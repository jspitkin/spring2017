import nltk
import re
import ssl
from nltk.parse.stanford import StanfordDependencyParser
from nltk.stem.porter import *

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

    print('noun phrases:', noun_phrases)
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


def lexical_coverage(sentence, noun_phrase, question):
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
