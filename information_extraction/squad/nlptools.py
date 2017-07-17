import nltk
import re
import ssl
import numpy as np
from nltk.parse.stanford import StanfordDependencyParser
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tag.stanford import StanfordNERTagger

def download_nltk_resources():
    """There is a known bug currently with nltk and ssl verification. This is a workaround via stackoverflow."""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download()


def dependency_tree(sentence):
    """"Returns the dependency tree for a sentence."""
    path_to_jar = "./lib/stanford-parser.jar"
    path_to_models_jar = "./lib/stanford-parser-3.7.0-models.jar"
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    tree = dependency_parser.raw_parse(sentence)
    for entry in tree:
        print(entry)
    return list(tree.__next__().triples())


def split_to_sentences(corpus):
    """Takes in a corpus and returns a list of sentences."""
    return nltk.tokenize.sent_tokenize(corpus)


def get_noun_phrases(sentence):
    noun_phrases = []
    tokens = tokenize(sentence)
    tokens_with_pos_tag = pos_tag_tokens(tokens)
    grammar = """
        NP: {<DT|PP\$>?<JJ>*<NN>}
            {<NNP>+}
            {<NN>+}
            {<PRP><NN>}
            {<PRP$>}
            {<NNP><NNS>}
            {<DT><NNP>}
            {<DT><CD><NNS>}
            {<CD><NNS>}
            {<NNS>+}
            {(<DT>?<RB>?)?<JJ|CD>*(<JJ|CD><,>)*<NN.*>+}
            {(<DT|PRP.>?<RB>?)?<JJ|CD>*(<JJ|CD><,>)*(<NN.*>)+}
            {<WP>}
            {<NNP><POS><NNP>}
        """
    chunker = nltk.RegexpParser(grammar)
    result = chunker.parse(tokens_with_pos_tag)

    for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
        noun_phrase = ""
        for word in subtree.leaves():
            noun_phrase += word[0] + " "
        noun_phrases.append(noun_phrase.strip())

    # Add all dates in the format dd/mm/yy as noun phrases
    match = re.findall(r'(\d+/\d+/\d+)', sentence)
    for m in match:
        noun_phrases.append(m)

    return noun_phrases


def longest_noun_phrase(context):
    """Given an example, returns the longest noun phrase"""
    longest_noun_phrase = ""
    sentences = split_to_sentences(context['paragraph'])
    for sentence in sentences:
        noun_phrases = get_noun_phrases(sentence)
        for np in noun_phrases:
            if len(np) > len(longest_noun_phrase):
                longest_noun_phrase = np
    return longest_noun_phrase


def lexical_overlap(sentence, noun_phrase, question):
    sentence_tokens = tokenize(sentence)
    noun_phrase_tokens = tokenize(noun_phrase)
    question_tokens = tokenize(question)

    sentence_tokens = [token.lower() for token in sentence_tokens]
    noun_phrase_tokens = [token.lower() for token in noun_phrase_tokens]
    question_tokens = [token.lower() for token in question_tokens]

    noun_phrase_context = list(set(sentence_tokens) - set(noun_phrase_tokens))
    noun_phrase_context = remove_common_words(noun_phrase_context)
    question_tokens = remove_common_words(question_tokens)

    return len(set(sentence_tokens).intersection(set(question_tokens)))


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def pos_tag_tokens(tokens):
    return nltk.pos_tag(tokens)


def remove_common_words(tokens):
    common_words = set(['a', 'an', 'the', 'in'])
    return list(set(tokens) - set(common_words))


def overlapping_words(sent_one, sent_two):
    bad_tokens = set(['.', ',', 'a', '\'s'])
    sent_one_tokenized = [word.lower() for word in nltk.word_tokenize(sent_one)]
    sent_two_tokenized = [word.lower() for word in nltk.word_tokenize(sent_two)]
    sent_two_tokenized = set(sent_two_tokenized)
    return [word for word in sent_one_tokenized if word in sent_two_tokenized and word not in bad_tokens]

def overlapping_words_minus_candidate(question, sentence, candidate):
    bad_tokens = set(['.', ',', 'a', '\'s'])
    candidate_tokenized = set([word.lower() for word in nltk.word_tokenize(candidate['phrase'])])
    question_tokenized = [word.lower() for word in nltk.word_tokenize(question)]
    sentence_tokenized = [word.lower() for word in nltk.word_tokenize(sentence)]
    sentence_tokenized = set(sentence_tokenized)
    return [word for word in question_tokenized if word in sentence_tokenized and word not in bad_tokens and word not in candidate_tokenized]


def set_question_features(candidates, sentences, vectorizer, tfidf_matrix):
    for candidate in candidates:
        candidate['lex-overlap'] = lexical_overlap(sentences[candidate['sentence_index']], candidate['phrase'], candidate['question'])
        candidate['tfidf-whole-sentence'] = 0
        for word in overlapping_words(candidate['question'], sentences[candidate['sentence_index']]):
            tfidf_score = get_tfidf_score(vectorizer, tfidf_matrix, word, candidate['context_key'])
            candidate['tfidf-whole-sentence'] += tfidf_score
        candidate['tfidf-minus-candidate'] = 0
        for word in overlapping_words_minus_candidate(candidate['question'], sentences[candidate['sentence_index']], candidate):
            tfidf_score = get_tfidf_score(vectorizer, tfidf_matrix, word, candidate['context_key'])
            candidate['tfidf-minus-candidate'] += tfidf_score
        candidate['tfidf-candidate'] = 0
        for word in overlapping_words(candidate['question'], candidate['phrase']):
            tfidf_score = get_tfidf_score(vectorizer, tfidf_matrix, word, candidate['context_key'])
            candidate['tfidf-candidate'] += tfidf_score
        candidate['tfidf-span'] = 0
        for word in candidate['phrase'].split():
            candidate['tfidf-span'] += get_tfidf_score(vectorizer, tfidf_matrix, word, candidate['context_key'])
                    

def set_w_word_features(candidates):
    # Set Boolean feature for the first 'w-word' found in the question or none if one isn't found.
    for candidate in candidates:
        # Set a contains a capitalized word feature as well
        candidate['contains-capital'] = 0
        for word in candidate['phrase']:
            if word[0].isupper():
                candidate['contains-capital'] = 1
        # Set a contains a digit feature
        candidate['w-word-who'] = 0
        candidate['w-word-what'] = 0
        candidate['w-word-where'] = 0
        candidate['w-word-when'] = 0
        candidate['w-word-which'] = 0
        candidate['w-word-none'] = 0
        found_w_word = False
        question = [word.lower() for word in candidate['question'].split()]
        for word in question:
            if word.lower() == 'who':
                candidate['w-word-who'] = 1
                found_w_word = True
                break
            elif word.lower() == 'what':
                candidate['w-word-what'] = 1
                found_w_word = True
                break
            elif word.lower() == 'where':
                candidate['w-word-where'] = 1
                found_w_word = True
                break
            elif word.lower() == 'when':
                candidate['w-word-when'] = 1
                found_w_word = True
                break
            elif word.lower() == 'which':
                candidate['w-word-which'] = 1
                found_w_word = True
                break
        if not found_w_word:
            candidate['w-word-none'] = 0

def set_context_features(candidates, sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentences.append(tokenize(sentence))
    for candidate in candidates:
        containing_sentence = tokenized_sentences[candidate['sentence_index']]
        candidate['sent-len'] = len(containing_sentence)
        candidate['length'] = len(candidate['phrase'].split())
        previous_token = containing_sentence[0]
        index = 0
        for token in containing_sentence[1:]:
            if token == candidate['phrase'].split()[0]:
                candidate['left-sent-len'] = index
                candidate['prev-word'] = previous_token
            if previous_token == candidate['phrase'].split()[-1]:
                candidate['right-sent-len'] = len(containing_sentence) - index - 1
                candidate['next-word'] = token
            previous_token = token
            index += 1
        if 'left-sent-len' not in candidate:
            candidate['left-sent-len'] = 0
        if 'right-sent-len' not in candidate:
            candidate['right-sent-len'] = 0


def get_tfidf_vectors(corpora):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpora)
    return tfidf_matrix, vectorizer


def get_tfidf_score(vectorizer, matrix, word, context_key):
    replace_list = ['.', '/', '-', ',']
    for entry in replace_list:
        word.replace(entry, ' ')
    dense = matrix.todense()
    context = dense[context_key].tolist()[0]
    score_sum = 0
    for w in word.split():
        if word in vectorizer.vocabulary_:
            word_key = vectorizer.vocabulary_[word]
            score_sum += context[word_key]
    return score_sum


def cull_candidate_list(candidate_list):
    # Remove any candidate noun phrases unlikely to be an answer
    bad_candidates = set(['.', 'a', 'an', 'the', 'he', 'she', 'it', 'them', 'mr.', 'mr', 'mrs.', 'mrs', 'who'])
    new_candidate_list = []
    for entry in candidate_list:
        entry_words = entry.split()
        new_entry = ""
        for entry_word in entry_words:
            if entry_word not in bad_candidates:
                new_entry = new_entry + entry_word + " "
        entry = new_entry.strip()
        if entry != '':
            new_candidate_list.append(entry)

    return new_candidate_list

def set_ner_features(candidates):
    path_to_model = "./lib/english.all.3class.nodistsim.crf.ser.gz"
    path_to_jar = "./lib/stanford-ner.jar"
    st = StanfordNERTagger(path_to_model, path_to_jar)
    for candidate in candidates:
        ner_tags = st.tag(tokenize(candidate['phrase']))
        candidate['is-other'] = 1
        candidate['is-location'] = 0
        candidate['is-organization'] = 0
        candidate['is-person'] = 0
        for tag in ner_tags:
            if tag[1] == 'LOCATION':
                candidate['is-location'] = 1
                candidate['is-other'] = 0
                break
            if tag[1] == 'ORGANIZATION':
                candidate['is-organization'] = 1
                candidate['is-other'] = 0
                break
            if tag[1] == 'PERSON':
                candidate['is-person'] = 1
                candidate['is-other'] = 0
                break
            
