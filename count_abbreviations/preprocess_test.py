import os
import sys
import pickle
import argparse

if __name__ == "__main__":
    from pathlib import Path
    script_folder = str(Path(__file__).resolve().parents[0])
    os.chdir(script_folder)

    parser = argparse.ArgumentParser(description="Preprocess test data.")
    parser.add_argument("--text", type=str, help="Text data file", required=True, dest="text")
    parser.add_argument("--var", type=str, help="Variation data file", required=True, dest="var")
    parser.add_argument("--train", type=str, help="Train data", required=True, dest="train")
    parser.add_argument("--out", type=str, help="Target test data", required=True, dest="out")
    args = parser.parse_args()

    if not os.path.exists(args.text):
        print("Cannot find text data at {}".format(args.text))
        sys.exit()
    if not os.path.exists(args.var):
        print("Cannot find variation data at {}".format(args.var))
        sys.exit()
    if not os.path.exists(args.train):
        print("Cannot find train data at {}".format(args.train))
        sys.exit()

    import preprocess_train

    print("Reading text...")
    text_data = preprocess_train.read_text_file(args.text)
    print("  Number of samples = {:d}".format(len(text_data)))

    print("Reading variations...")
    var_data = preprocess_train.read_variations_file(args.var)
    print("  Number of samples = {:d}".format(len(var_data)))

    assert len(text_data) == len(var_data)

    print('Read train data (we need dict)...')
    train_obj = pickle.load(open(args.train, 'rb'))
    dictionary = train_obj['dictionary']
    dictionary_freqs = train_obj['dictionary_freqs']

    print("Compose test data...")
    x, y = preprocess_train.compose_numpy_data(text_data, var_data, dictionary)

    print("Pickling data...")
    obj = {"test_x": x, "test_y": y}
    with open(args.out, 'wb') as fp:
        pickle.dump(obj, fp)
    print('  Saved to {}'.format(args.out))

    print("Done.")