#!/usr/bin/env python
# -*- coding: utf-8 -*-
# データセットを学習セットとテストセットに分割する

import argparse
import importlib
import random
import shutil

metadata = importlib.import_module(".metadata", "free-spoken-digit-dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, required=True, help="秘密のシード")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    training = open("./training.csv", "w")
    test = open("./test.csv", "w")
    test_truth = open("./test_truth.csv", "w")

    training.write("path,label\n")
    test.write("path\n")
    test_truth.write("path,label\n")

    speaker = list(metadata.metadata.keys())

    for name in speaker:
        training_index = 0
        test_index = 0

        for digit in random.sample(range(10), 10):
            shuffled = random.sample(range(50), 50)
            for original_index in shuffled[:5]:
                from_filepath = "{directory}/{label}_{name}_{index}.wav".format(
                    directory="free-spoken-digit-dataset/recordings",
                    label=digit,
                    name=name,
                    index=original_index,
                )
                to_filepath = "{directory}/{name}_{index}.wav".format(
                    directory="dataset/test",
                    name=name,
                    index=test_index,
                )
                shutil.copy2(from_filepath, to_filepath)
                test.write(
                    "{to_filepath}\n".format(
                        to_filepath=to_filepath,
                    )
                )
                test_truth.write(
                    "{to_filepath},{label}\n".format(
                        to_filepath=to_filepath,
                        label=digit,
                    )
                )
                test_index += 1
            for original_index in shuffled[5:50]:
                from_filepath = "{directory}/{label}_{name}_{index}.wav".format(
                    directory="free-spoken-digit-dataset/recordings",
                    label=digit,
                    name=name,
                    index=original_index,
                )
                to_filepath = "{directory}/{name}_{index}.wav".format(
                    directory="dataset/train",
                    name=name,
                    index=training_index,
                )
                shutil.copy2(from_filepath, to_filepath)
                training.write(
                    "{directory}/{name}_{index}.wav,{label}\n".format(
                        directory="dataset/train",
                        name=name,
                        index=training_index,
                        label=digit,
                    )
                )
                training_index += 1

    training.close()
    test.close()


if __name__ == "__main__":
    main()
