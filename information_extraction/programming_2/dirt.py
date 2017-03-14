# Author: Jake Pitkin
# Last Update: Feb 28 2017
import sys
import math

def main():
    'Program entry point'
    if len(sys.argv) != 4:
        print_usage()

    corpus_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    min_freq = int(sys.argv[3])

    corpus_entries = parse_corpus_file(corpus_file_path)
    triple_database = get_triple_database(corpus_entries)
    legal_triple_database = get_legal_triple_database(triple_database)
    filtered_database = get_minfreq_filtered_triple_database(legal_triple_database, min_freq)
    distinct_paths = get_distinct_paths(filtered_database)

    print("Found", get_distinct_path_count(legal_triple_database), "distinct paths,", get_distinct_path_count(filtered_database), "after minfreq filtering.")
    print("Found", len(legal_triple_database), "path instances,", len(filtered_database), "after minfreq filtering.")

    for test_phrase in parse_test_file(test_file_path):
        if test_phrase not in distinct_paths:
            print()
            print("MOST SIMILAR RULES FOR:", test_phrase)
            print("This phrase is not in the triple database.")
            continue

        print()
        print("MOST SIMILAR RULES FOR:", test_phrase)
        index = 1
        previous_score = 0

        for rule in get_similar_rules(test_phrase, filtered_database, distinct_paths):
            if index > 5 and previous_score != rule['score'] or rule['score'] == 0:
                break
            print(index, ". ", rule['path'], '\t\t', rule['score'], sep='')
            index += 1
            previous_score = rule['score']


def mutual_information(phrase, slot, head_noun, database):
    head_noun_for_path_count = 0
    total_head_nouns = 0
    path_for_slot_count = 0
    head_noun_for_slot_count = 0

    unique_fillers = []
    unique_fillers_in_slot = []

    for entry in database:
        if entry['path'] == phrase and entry[slot] == head_noun:
            head_noun_for_path_count += 1
        if entry['path'] == phrase:
            unique_fillers_in_slot.append(entry[slot])
        if entry[slot] == head_noun:
            head_noun_for_slot_count += 1
        unique_fillers.append(entry[slot])
    total_head_nouns = len(unique_fillers)
    path_for_slot_count = len(unique_fillers_in_slot)

    numerator = head_noun_for_path_count * total_head_nouns
    denominator = path_for_slot_count * head_noun_for_slot_count

    if numerator == 0 or denominator == 0:
        return 0

    result = math.log(numerator/denominator, 2)

    if result < 0:
        return 0

    return result


def sim(slot, test_path, path, database):
    test_path_head_nouns = set()
    path_head_nouns = set()
    in_both_paths = set()
    for entry in database:
        if entry['path'] == test_path:
            test_path_head_nouns.add(entry[slot])
        if entry['path'] == path:
            path_head_nouns.add(entry[slot])
    in_both_paths = set.intersection(test_path_head_nouns, path_head_nouns)
    test_path_head_nouns = list(test_path_head_nouns)
    path_head_nouns = list(path_head_nouns)
    in_both_paths = list(in_both_paths)

    numerator = 0
    denominator = 0

    for word in in_both_paths:
        numerator += mutual_information(test_path, slot, word, database)
        numerator += mutual_information(path, slot, word, database)
    for word in test_path_head_nouns:
        denominator += mutual_information(test_path, slot, word, database)
    for word in path_head_nouns:
        denominator += mutual_information(path, slot, word, database)

    if denominator == 0:
        return 0

    if numerator == denominator:
        return 1

    return numerator / denominator


def path_similarity(test_path, path, database):
    sim_x = sim("slot_x", test_path, path, database)
    sim_y = sim("slot_y", test_path, path, database)
    return math.sqrt(sim_x * sim_y)


def get_similar_rules(test_phrase, database, distinct_paths):
    scores = []
    for path in distinct_paths:
        if test_phrase == path:
            scores.append({'score':1, 'path':path})
        else:
            score = path_similarity(test_phrase, path, database)
            scores.append({'score':score, 'path':path})
    scores = sorted(scores, key=lambda k: k['score'], reverse=True)
    return scores


def get_triple_database(corpus_entries):
    triple_database = []
    previous_entry = corpus_entries[0]
    database_entry = {}
    processing_entry = False
    for entry in corpus_entries[1:]:
        # End of sentence - clear possible database entry
        if entry['phrase'][0] == "<eos":
            processing_entry = False
            database_entry = {}
        # Processing a database entry and there is more to append
        elif processing_entry and entry['syntactic_type'] != "NP":
            for word in entry['phrase']:
                database_entry['path'].append(word)
        # End of a database entry
        elif processing_entry and entry['syntactic_type'] == "NP":
            database_entry['slot_y'] = entry['phrase'][-1]
            database_entry['path'] = ' '.join(database_entry['path'])
            triple_database.append(database_entry)
            database_entry = {}
            processing_entry = False
        # Start of a new database entry
        elif previous_entry['syntactic_type'] == "NP" and entry['syntactic_type'] != "NP":
            database_entry['slot_x'] = previous_entry['phrase'][-1]
            database_entry['path'] = []
            for word in entry['phrase']:
                database_entry['path'].append(word)
            processing_entry = True
        previous_entry = entry
    return triple_database


def equal_entries(e1, e2):
    if e1['slot_x'] == e2['slot_x'] and e1['path'] == e2['path'] and e1['slot_y'] == e2['slot_y']:
        return True
    return False

def get_distinct_path_count(triple_database):
    distinct_paths = set()
    for entry in triple_database:
        distinct_paths.add(entry['path'])
    return len(distinct_paths)

def get_distinct_paths(triple_database):
    distinct_paths = set()
    for entry in triple_database:
        distinct_paths.add(entry['path'])
    return list(distinct_paths)


def get_legal_triple_database(triple_database):
    legal_triple_database = []
    illegal_paths = set(["is", "are", "be", "was", "were", "said", "have", "has", "had", "and", "or",
                         ">comma", ">squote", ">rparen", ">lparen", ">period", ">minus", ">ampersand"])
    for entry in triple_database:
        if len(entry['path'].split()) > 1:
            legal_triple_database.append(entry)
        elif len(entry['path'].split()) == 1 and entry['path'] not in illegal_paths:
            legal_triple_database.append(entry)
    return legal_triple_database


def get_minfreq_filtered_triple_database(triple_database, minfreq):
    path_counts = {}
    filtered_triple_database = []
    for entry in triple_database:
        if entry['path'] in path_counts:
            path_counts[entry['path']] += 1
        else:
            path_counts[entry['path']] = 1
    for entry in triple_database:
        if path_counts[entry['path']] >= minfreq:
            filtered_triple_database.append(entry)
    return filtered_triple_database

def parse_corpus_file(path):
    corpus_entries = []
    with open(path) as input_file:
        for line in input_file:
            line = line.split()
            if len(line) >= 3:
                syntactic_type = line[0]
                phrase = line[2:]
                phrase = [x.lower() for x in phrase]
                corpus_entries.append({'syntactic_type' : syntactic_type, 'phrase' : phrase})
    return corpus_entries

def parse_test_file(path):
    test_words = []
    with open(path) as input_file:
        for line in input_file:
            test_words.append(line.strip().lower())
    return test_words

def print_usage():
    print("usage: python3 dirt.py <corpus_file> <test_file> <minfreq>")

if __name__ == '__main__':
    main()
