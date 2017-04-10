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

    # 30 questions based on the Super Bowl 50 Corpus
    examples = []
    for example in dev['examples']:
        if example['context_key'] == 0:
            examples.append(example)
    corpus = dev['contexts'][0]['paragraph']

    # Extract the candidate answers from the corpus
    # Heuristic - assume the answer is a noun phrase in the parse tree
    candidates = []
    sentences = nlptools.split_to_sentences(corpus)
    for index, sentence in enumerate(sentences):
        nps = nlptools.get_noun_phrases(sentence)
        candidates.extend([{'phrase':np, 'sentence_index':index} for np in nps])

    nlptools.set_context_features(candidates, sentences)

    for candidate in candidates:
        print(candidate)
        break

    print(corpus)


def print_usage():
    print("usage: python3 main.py <training-set-path> <dev-set-path> <predictions-path>")


if __name__ == '__main__':
    main()
