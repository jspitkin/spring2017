import ioutil
import logistic_regression as lr
import nlptools
import sys
import re
from random import shuffle
from random import randint

vectorizer = None
tfidf_matrix = None

def main():
    'Program entry point'
    global vectorizer, tfidf_matrix

    if len(sys.argv) != 4:
        print_usage();
        return -1

    train_set_path = sys.argv[1]
    dev_set_path = sys.argv[2]
    prediction_path = sys.argv[3]

    train = ioutil.read_examples_from_file(train_set_path)
    dev = ioutil.read_examples_from_file(dev_set_path)

    # Split the training examples based on context
    examples = {}
    corpora = {}
    for example in train['examples']:
        if example['context_key'] not in examples:
            examples[example['context_key']] = []
        else:
            examples[example['context_key']].append(example)
    context_keys = sorted(examples.keys())
    for key in context_keys:
        corpora[key] = train['contexts'][key]['paragraph']

    sentences = nlptools.split_to_sentences(article)
    total_nps = []
    for sentence in sentences:
        nps = nlptools.get_noun_phrases(sentence)
        nps = nlptools.remove_common_words(nps)
        nps = nlptools.cull_candidate_list(nps)
        total_nps.extend(nps)
    print(total_nps)

    tfidf_matrix, vectorizer  = nlptools.get_tfidf_vectors(list(corpora.values()))

    # For each context, create feature vectors for each relevant example
    training_vectors = []
    for key, context in corpora.items():
        print('training:', key)
        training_vectors.extend(feature_vectors_for_context(key, context, examples))

    for training_vector in training_vectors:
        del training_vector[0]

    # Write the feature vectors to file
    ioutil.write_feature_vectors('train_vectors.txt', training_vectors)

    # Train a logistic regression model
    model = lr.train_even()

    # Make the predictions for each development example and write to file
    make_predictions(model, dev, prediction_path)


def make_predictions(model, dev, prediction_path):
    predictions = {}
    # Split the development examples based on context
    examples = {}
    corpora = {}
    for example in dev['examples']:
        if example['context_key'] not in examples:
            examples[example['context_key']] = [example]
        else:
            examples[example['context_key']].append(example)
    context_keys = sorted(examples.keys())
    for key in context_keys:
        corpora[key] = dev['contexts'][key]['paragraph']
    
    total_examples = 0
    correct_answer_in_candidates = 0
    exact_match = 0
    partial_match = 0
    # For each example, create feature vectors for each candidate answer
    for key, example_group in examples.items():
        print('prediciton:', key)
        total_examples += len(example_group)
        for example in example_group:
            context = corpora[key]
            candidate_vectors = feature_vectors_for_example(context, example)

            possible_answers = set([answer['text'] for answer in example['possible_answers']])
            for candidate_vector in candidate_vectors:
                if candidate_vector[0] in possible_answers:
                    correct_answer_in_candidates += 1
                    break

            # Evaluate each candidate against the logistic regression model
            best_candidate = ''
            best_confidence_score = 0
            for index, candidate_vector in enumerate(candidate_vectors):
                candidate_phrase = candidate_vector[0]
                # Pop off the label and the phrase
                candidate_vector.pop(0)
                candidate_vector.pop(0)
                prediction = model.predict_proba([candidate_vector])
                if prediction[0][1] > best_confidence_score:
                    best_confidence_score = prediction[0][1]
                    best_candidate = candidate_phrase

            found_exact_match = False
            for possible_answer in possible_answers:
                if best_candidate == possible_answer:
                    exact_match += 1
                    found_exact_match = True
                    break

            for possible_answer in possible_answers:
                if not found_exact_match and len(set.intersection(set(possible_answer.split()), set(best_candidate.split()))) > 0:
                    partial_match += 1
                    break

            # Make prediction with the candidate with the highest score
            predictions[example['id']] = best_candidate
    print('found answer:', correct_answer_in_candidates, 'example count:', total_examples)
    print('exact matches:', exact_match, 'partial matches:', partial_match) 
    print()

    # Write the predictions to file
    ioutil.write_prediction_file(prediction_path, predictions)


def feature_vectors_for_example(context, example):
    global vectorizer
    global tfidf_matrix

    sentences = nlptools.split_to_sentences(context)
    sentences_indexed = []
    start_index = 0
    for sentence in sentences:
        sentences_indexed.append({'sentence': sentence, 'start_index':start_index})
        start_index += len(sentence)

    # Extract the candidate answers from the corpus
    # Heuristic - assume the answer is a noun phrase in the parse tree
    candidates = []
    for index, sentence in enumerate(sentences):
        nps = nlptools.get_noun_phrases(sentence)
        candidates.extend([{'phrase':np, 'sentence_index':index, 'question':example['question'], 'context_key':example['context_key'], 'label':0} for np in nps])
    candidates = nlptools.cull_candidate_list(candidates)

    # Create features for each of the candidates based on sentence context
    nlptools.set_context_features(candidates, sentences)
    nlptools.set_question_features(candidates, sentences, vectorizer, tfidf_matrix)
    nlptools.set_w_word_features(candidates)

    # Get feature vector based on features of the candidates
    return get_feature_vectors(candidates)


def feature_vectors_for_context(key, context, examples):
    global vectorizer
    global tfidf_matrix

    sentences = nlptools.split_to_sentences(context)
    sentences_indexed = []
    start_index = 0
    for sentence in sentences:
        sentences_indexed.append({'sentence': sentence, 'start_index':start_index})
        start_index += len(sentence)

    # Extract the candidate answers from the corpus
    # Heuristic - assume the answer is a noun phrase in the parse tree
    candidates = []
    for index, sentence in enumerate(sentences):
        nps = nlptools.get_noun_phrases(sentence)
        candidates.extend([{'phrase':np, 'sentence_index':index, 'label': 0} for np in nps])
    candidates = nlptools.cull_candidate_list(candidates)

    # Create a copy of each candidate to create an example for each question
    extended_candidates = []
    for candidate in candidates:
        #for example in examples[key]:
            # If there is overlap with this candidate and the true answer, ignore this neg example
        example = examples[key][randint(0, len(examples[key]) - 1)]
        discard_candidate = False
        for answer in example['possible_answers']:
            if len(set.intersection(set(candidate['phrase'].split()), set(answer['text']))) > 0:
                discard_candidate = True
        if discard_candidate:
                continue
        new_candidate = candidate.copy()
        new_candidate['question'] = example['question']
        new_candidate['context_key'] = example['context_key']
        extended_candidates.append(new_candidate)

    # Create candidates from the golden labeled answers
    for example in examples[key]:
        for answer in example['possible_answers']:
            prev_sentence = sentences_indexed[0]
            for index, sentence in enumerate(sentences_indexed):
                if answer['answer_start'] < sentence['start_index']:
                    extended_candidates.append({'phrase': answer['text'], 'sentence_index': index-1, 'label': 1, 'question':example['question'], 'context_key':example['context_key']})
                    break
                prev_sentence = sentence

    # Create features for each of the candidates based on sentence context
    nlptools.set_context_features(extended_candidates, sentences)
    nlptools.set_question_features(extended_candidates, sentences, vectorizer, tfidf_matrix)
    nlptools.set_w_word_features(extended_candidates)

    # Get feature vector based on features of the candidates
    return get_feature_vectors(extended_candidates)


def get_feature_vectors(candidates):
    feature_vectors = [[candidate['phrase'], candidate['label']] for candidate in candidates]
    for index, candidate in enumerate(candidates):
        sorted(candidate)
        for key, value in candidate.items():
            if key != 'label' and key != 'phrase' and key != 'prev-word' and key != 'next-word' and key != 'question' and key != 'context_key':
                feature_vectors[index].append(value)
    return feature_vectors


def print_usage():
    print("usage: python3 main.py <training-set-path> <dev-set-path> <predictions-path>")


if __name__ == '__main__':
    main()
