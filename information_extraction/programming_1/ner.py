import sys

LABEL_LOOKUP_TABLE = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-LOC': 3,
    'I-LOC': 4,
    'B-ORG': 5,
    'I-ORG': 6
}


def main():
    'Program Entry Point'
    if len(sys.argv) != 4:
        print_usage()
        return -1

    training_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    ftype = sys.argv[3]

    # Training examples
    training_examples = parse_input_file(training_data_path)
    possible_features = generate_possible_features(training_examples, ftype)
    write_features_to_file(training_examples, possible_features, training_data_path, ftype)

    # Test examples
    test_examples = parse_input_file(test_data_path)
    write_features_to_file(test_examples, possible_features, test_data_path, ftype)

    unique_words = len(get_unique_words(training_examples)) - 3
    unique_pos = len(get_unique_pos_tags(training_examples)) - 3

    print_debug_information(len(training_examples), len(test_examples), unique_words, unique_pos)

def print_debug_information(training_count, test_count, word_count, pos_count):
    print("Found", training_count, "training instances with", word_count, "distinct words and", pos_count, "distinct POS tags.")
    print("Found", test_count, "test instances.")

def parse_input_file(path):
    examples = []
    beg_sent = True
    previous_example = {}
    with open(path) as input_file:
        for line in input_file:
            line = line.split()
            if len(line) == 3:
                example = {'pos': line[1], 'word': line[2], 'beg_sent': beg_sent, 'end_sent': False}
                example['label'] = LABEL_LOOKUP_TABLE[line[0]]
                previous_example = example
                examples.append(example)
                beg_sent = False
            else:
                beg_sent = True
                previous_example['end_sent'] = True
    previous_example['end_sent'] = True
    return examples


def get_unique_words(examples):
    unique_words = set(['PHI', 'OMEGA', 'UNKWORD'])
    for example in examples:
        unique_words.add(example['word'])
    return list(unique_words)


def get_unique_pos_tags(examples):
    unique_pos_tags = set(['PHIPOS', 'OMEGAPOS', 'UNKPOS'])
    for example in examples:
        unique_pos_tags.add(example['pos'])
    return list(unique_pos_tags)


def generate_feature_vectors(examples, possible_features, ftype):
    features_vectors = []
    for index, example in enumerate(examples):
        feature_vector = {'label': example['label'], 'features': []}
        # Current word features
        if 'curr-' + example['word'] in possible_features:
            feature_vector['features'].append(possible_features['curr-' + example['word']])
        else:
            feature_vector['features'].append(possible_features['curr-UNKWORD'])
        # Capitalized features
        if ftype != 'word':
            if len(example['word']) > 0 and (example['word'])[0].isupper():
                feature_vector['features'].append(possible_features['capitalized'])
        # Part of speech features
        if ftype == 'poscon' or ftype == 'bothcon':
            if example['beg_sent']:
                feature_vector['features'].append(possible_features['prevpos-PHIPOS'])
            elif index != 0 and 'prevpos-' + examples[index-1]['pos'] in possible_features:
                feature_vector['features'].append(possible_features['prevpos-' + examples[index-1]['pos']])
            else:
                feature_vector['features'].append(possible_features['prevpos-UNKPOS'])
            if example['end_sent']:
                feature_vector['features'].append(possible_features['nextpos-OMEGAPOS'])
            elif index != len(examples)-1 and 'nextpos-' + examples[index+1]['pos'] in possible_features:
                feature_vector['features'].append(possible_features['nextpos-' + examples[index+1]['pos']])
            else:
                feature_vector['features'].append(possible_features['nextpos-UNKPOS'])
        # Context features
        if ftype == 'lexcon' or ftype == 'bothcon':
            if example['beg_sent']:
                feature_vector['features'].append(possible_features['prev-PHI'])
            elif index != 0 and 'prev-' + examples[index-1]['word'] in possible_features:
                feature_vector['features'].append(possible_features['prev-' + examples[index-1]['word']])
            else:
                feature_vector['features'].append(possible_features['prev-UNKWORD'])
            if example['end_sent']:
                feature_vector['features'].append(possible_features['next-OMEGA'])
            elif index != len(examples)-1 and 'next-' + examples[index+1]['word'] in possible_features:
                feature_vector['features'].append(possible_features['next-' + examples[index+1]['word']])
            else:
                feature_vector['features'].append(possible_features['next-UNKWORD'])
        features_vectors.append(feature_vector)
    return features_vectors


def write_features_to_file(examples, possible_features, input_path, ftype):
    feature_vectors = generate_feature_vectors(examples, possible_features, ftype)
    path = input_path + '.' + ftype
    f = open(path, 'w')
    for feature_vector in feature_vectors:
        features = feature_vector['features']
        features.sort()
        feature_line = str(feature_vector['label'])
        for feature in features:
            feature_line += ' ' + str(feature) + ':1'
        f.write(feature_line + '\n')
    f.close()


def generate_possible_features(examples, ftype):
    unique_words = get_unique_words(examples)
    unique_pos_tags = get_unique_pos_tags(examples)
    current_id = 1
    features = {}
    for unique_word in unique_words:
        features['curr-' + unique_word] = current_id
        current_id += 1
        if ftype == 'bothcon' or ftype == 'lexcon':	
            features['prev-' + unique_word] = current_id
            current_id += 1
            features['next-' + unique_word] = current_id
            current_id += 1
    for unique_pos_tag in unique_pos_tags:
        if ftype == 'bothcon' or ftype == 'poscon':
            features['prevpos-' + unique_pos_tag] = current_id
            current_id += 1
            features['nextpos-' + unique_pos_tag] = current_id
            current_id += 1
    if ftype != 'word':
        features['capitalized'] = current_id
    return features


def print_usage():
    print("usage: python3 ner.py <train_file> <test_file> <ftype>")

if __name__ == '__main__':
    main()
