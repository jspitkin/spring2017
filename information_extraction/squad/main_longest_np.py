import ioutil
import nlptools
import sys

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
    longest_np_per_context = {}

    progress = 0
    for context_key, context in dev['contexts'].items():
        longest_np_per_context[context['key']] = nlptools.longest_noun_phrase(context)
        print(progress / len(dev['contexts'].items()))
        progress += 1

    for example in dev['examples']:
        predictions[example['id']] = longest_np_per_context[example['context_key']]

    ioutil.write_prediction_file(prediction_path, predictions)


def print_usage():
    print("usage: python3 main.py <training-set-path> <dev-set-path> <predictions-path>")


if __name__ == '__main__':
    main()
