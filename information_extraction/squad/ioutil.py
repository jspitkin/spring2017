import json

def read_examples_from_file(path):
    examples = []
    contexts = {}
    context_key = 0
    with open(path, 'r') as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
        for catagory in dataset:
            for paragraph in catagory['paragraphs']:
                contexts[context_key] = {'paragraph':paragraph['context'], 'title':catagory['title'], 'key':context_key}
                for qas in paragraph['qas']:
                    example = {}
                    example['title'] = catagory['title']
                    example['possible_answers'] = []
                    example['context_key'] = context_key
                    example['question'] = qas['question']
                    example['id'] = qas['id']
                    for answer in qas['answers']:
                        example['possible_answers'].append(answer)
                    examples.append(example)
                context_key += 1
    return {'examples':examples, 'contexts':contexts}

def write_feature_vectors(path, candidates):
    with open(path, 'w') as feature_vector_file:
        for candidate in candidates:
            feature_vector = []
            feature_vector.append(candidate['label'])
            feature_vector.append(candidate['sent-len'])
            feature_vector.append(candidate['length'])
            feature_vector.append(candidate['left-sent-len'])
            feature_vector.append(candidate['right-sent-len'])
            for entry in feature_vector:
                feature_vector_file.write(str(entry))
                feature_vector_file.write(' ')
            feature_vector_file.write('\n')
        

def write_prediction_file(path, predictions):
    with open(path, 'w') as predictions_file:
        json.dump(predictions, predictions_file)
