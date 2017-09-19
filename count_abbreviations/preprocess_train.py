import os
import sys
import numpy
import pickle
import argparse
import operator

NUM_FEATURES = 10000


def is_interesting_word(word):  # Silly heuristics to filter terms.
    if word[1:].lower() != word[1:]:
        return True
    return False


def read_text(text):
    clear_text = text
    for char in [".", ",", ":", ";", "!", "?", "(", ")", "[", "]"]:
        clear_text = clear_text.replace(char, " ")
    words = clear_text.split(" ")
    result = []
    for word in words:
        if is_interesting_word(word):
            result.append(word.lower())
    return result


def read_text_file(path):
    result = []
    with open(path) as fp:
        for line in fp:
            line = line.rstrip()
            items = line.split("||")
            if len(items) != 2:
                continue
            id = int(items[0])
            text = items[1]
            interesting_words = read_text(text)
            result.append((id, interesting_words))
    return result


def read_variations_file(path):
    result = []
    lines = open(path).readlines()
    for i in range(1, len(lines)):
        line = lines[i]
        items = line.strip().split(",")
        id = int(items[0])
        gene = items[1]
        var = items[2]
        label = None
        if len(items) > 3:
            label = items[3]
        result.append((id, gene, var, label))
    return result


def compose_dictionary(text_data):
    freqs = dict()
    for id, words in text_data:
        for word in words:
            if word not in freqs:
                freqs[word] = 0
            freqs[word] += 1
    result = []
    for key, val in freqs.items():
        result.append((key, val))
    result = sorted(result, key=operator.itemgetter(1), reverse=True)
    result = result[:NUM_FEATURES]
    dictionary, dictionary_freqs = zip(*result)
    return dictionary, dictionary_freqs


def build_word2idx(dictionary):
    word2idx = dict()
    for idx in range(len(dictionary)):
        word2idx[dictionary[idx]] = idx
    return word2idx


def compose_numpy_data(text_data, var_data, dictionary):
    word2idx = build_word2idx(dictionary)

    num_samples = len(text_data)
    x = numpy.zeros((num_samples, NUM_FEATURES), dtype="float32")
    y = numpy.zeros(num_samples, dtype="int32")

    for i in range(num_samples):
        words = text_data[i][1]
        for word in words:
            if word in word2idx:
                idx = word2idx[word]
                x[i, idx] += 1.0
        label = var_data[i][3]
        if label is not None:
            y[i] = label
        else:
            y[i] = -1

    # Sanity check.
    sum_x = x.sum(axis=1)
    pos_count = numpy.count_nonzero(sum_x > 0)
    print("  Number of good samples = {:d}".format(pos_count))

    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts interesting words from text.")
    parser.add_argument("--text", type=str, help="Text data file", required=True, dest="text")
    parser.add_argument("--var", type=str, help="Variation data file", required=True, dest="var")
    parser.add_argument("--dict", type=str, help="Target dictionary", required=True, dest="dict")
    args = parser.parse_args()

    if not os.path.exists(args.text):
        print("Cannot find text data at {}".format(args.text))
        sys.exit()
    if not os.path.exists(args.var):
        print("Cannot find variation data at {}".format(args.var))
        sys.exit()

    print("Reading text...")
    text_data = read_text_file(args.text)
    print("  Number of samples = {:d}".format(len(text_data)))

    print("Reading variations...")
    var_data = read_variations_file(args.var)
    print("  Number of samples = {:d}".format(len(var_data)))

    assert len(text_data) == len(var_data)

    print("Compose dictionary...")
    dictionary, dictionary_freqs = compose_dictionary(text_data)
    print('  Dictionary size = {:d}'.format(len(dictionary)))
    for i in range(len(dictionary)):
        print('  {:20} -> {:d}'.format(dictionary[i], dictionary_freqs[i]))

    print("Compose train data...")
    x, y = compose_numpy_data(text_data, var_data, dictionary)

    print("Pickling data...")
    obj = {
        "dictionary": dictionary,
        "dictionary_freqs": dictionary_freqs,
        "train_x": x,
        "train_y": y}
    with open(args.dict, 'wb') as fp:
        pickle.dump(obj, fp)
    print('  Saved to {}'.format(args.dict))

    print("Done.")
