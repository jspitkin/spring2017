import ioutil
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

    predictions = {}
    prediction_count = len(dev['examples'])
    progress = 0
    shuffled_examples = dev['examples']
    shuffle(shuffled_examples)
    for example in shuffled_examples:
        progress += 1
        print('progress:', (progress/prediction_count))
        best_lexical_overlap = 0
        best_answer = ""
        context_sentences = nlptools.split_to_sentences(dev['contexts'][example['context_key']]['paragraph'])
        for sentence in context_sentences:
            for noun_phrase in nlptools.get_noun_phrases(sentence):
                lexical_overlap = nlptools.lexical_coverage(sentence, noun_phrase, example['question'])
                if lexical_overlap > best_lexical_overlap:
                    best_lexical_overlap = lexical_overlap
                    best_answer = noun_phrase
        predictions[example['id']] = best_answer
        print('best answer:', best_answer)
        print('sentences:', context_sentences)
        print('overlap:', best_lexical_overlap)
        print('question:', example['question'])
        print('answer:', example['possible_answers'][0])
        break

    ioutil.write_prediction_file(prediction_path, predictions)


def print_usage():
    print("usage: python3 main.py <training-set-path> <dev-set-path> <predictions-path>")


if __name__ == '__main__':
    main()