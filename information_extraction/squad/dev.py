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

    train = ioutil.read_examples_from_file(train_set_path)
    dev = ioutil.read_examples_from_file(dev_set_path)

    # 30 questions based on the Super Bowl 50 Corpus
    examples = []
    for example in dev['examples']:
        if example['context_key'] == 0:
            examples.append(example)
    corpus = dev['contexts'][0]['paragraph']

    sentences = nlptools.split_to_sentences(corpus)
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
    for example in dev['examples']:
        if example['context_key'] != 0:
            continue
        for answer in example['possible_answers']:
            prev_sentence = sentences_indexed[0]
            for index, sentence in enumerate(sentences_indexed):
                if answer['answer_start'] < sentence['start_index']:
                    candidates.append({'phrase': answer['text'], 'sentence_index': index, 'label': 1})
                    break
                prev_sentence = sentence
    
    nlptools.set_context_features(candidates, sentences)
    training_vectors = nlptools.set_question_features(candidates, examples)

    # Write feature vectors to file
    ioutil.write_feature_vectors('train_vectors.txt', training_vectors)

    # Train a logistic regression model
    model = lr.train() 

    # Get a confidence score for each candidate
    candidates = []
    for index, sentence in enumerate(sentences):
        nps = nlptools.get_noun_phrases(sentence)
        candidates.extend([{'phrase':np, 'sentence_index':index, 'label': 0} for np in nps])

    example = examples[0]['question']
    print('question:', example)
    nlptools.set_context_features(candidates, sentences)
    candidate_vectors = nlptools.set_question_features(candidates, example)

    best_candidate = ''
    best_confidence_score = 0
    for candidate in candidates:
        feature_vector = []
        feature_vector.append(candidate['sent-len'])
        feature_vector.append(candidate['length'])
        feature_vector.append(candidate['left-sent-len'])
        feature_vector.append(candidate['right-sent-len'])
        prediction = model.predict_proba([feature_vector])
        if prediction[0][1] > best_confidence_score:
            best_confidence_score = prediction[0][1]
            best_candidate = candidate

    print(best_candidate['phrase'], best_confidence_score)


def print_usage():
    print("usage: python3 main.py <training-set-path> <dev-set-path> <predictions-path>")


if __name__ == '__main__':
    main()
