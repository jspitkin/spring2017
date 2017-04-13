import ioutil
import logistic_regression as lr
import nlptools
import sys
from random import shuffle

def main():
    'Program entry point'
    if len(sys.argv) != 4:
        print_usage();
        return -1

    train_set_path = sys.argv[1]
    dev_set_path = sys.argv[2]
    prediction_path = sys.argv[3]

    #train = ioutil.read_examples_from_file(train_set_path)
    train = ioutil.read_examples_from_file(dev_set_path)
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

    # For each context, create feature vectors for each relevant example
    training_vectors = []
    for key, context in corpora.items():
        x = 5
        #training_vectors.extend(feature_vectors_for_context(key, context, examples))

    # Write the feature vectors to file
    #ioutil.write_feature_vectors('train_vectors.txt', training_vectors)

    # Train a logistic regression model
    model = lr.train()

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

    print('examples:', len(examples.items()))
    # For each example, create feature vectors for each candidate answer
    for key, example_group in examples.items():
        print(key)
        for example in example_group:
            context = corpora[key]
            candidates, candidate_vectors = feature_vectors_for_example(context, example)

            # Evaluate each candidate against the logistic regression model
            best_candidate = ''
            best_confidence_score = 0
            for index, candidate_vector in enumerate(candidate_vectors):
                candidate_vector.pop(0)
                prediction = model.predict_proba([candidate_vector])
                if prediction[0][1] > best_confidence_score:
                    best_confidence_score = prediction[0][1]
                    best_candidate = candidates[index]['phrase']

            # Make prediction with the candidate with the highest score
            predictions[example['id']] = best_candidate

    # Write the predictions to file
    ioutil.write_prediction_file(prediction_path, predictions)


def feature_vectors_for_example(context, example):
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

    # Create features for each of the candidates based on sentence context
    nlptools.set_context_features(candidates, sentences)

    # Get feature vector based on features of the candidates
    return candidates, get_feature_vectors(candidates)


def feature_vectors_for_context(key, context, examples):
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

    # Create candidates from the golden labeled answers
    for example in examples[key]:
        for answer in example['possible_answers']:
            prev_sentence = sentences_indexed[0]
            for index, sentence in enumerate(sentences_indexed):
                if answer['answer_start'] < sentence['start_index']:
                    candidates.append({'phrase': answer['text'], 'sentence_index': index, 'label': 1})
                    break
                prev_sentence = sentence

    # Create features for each of the candidates based on sentence context
    nlptools.set_context_features(candidates, sentences)

    # Get feature vector based on features of the candidates
    return get_feature_vectors(candidates)


def get_feature_vectors(candidates):
    feature_vectors = [[candidate['label']] for candidate in candidates]
    for index, candidate in enumerate(candidates):
        sorted(candidate)
        for key, value in candidate.items():
            if key != 'label' and key != 'phrase' and key != 'prev-word' and key != 'next-word':
                feature_vectors[index].append(value)
    return feature_vectors


def print_usage():
    print("usage: python3 main.py <training-set-path> <dev-set-path> <predictions-path>")


if __name__ == '__main__':
    main()
