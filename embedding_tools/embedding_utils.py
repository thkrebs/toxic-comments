# Used to cleanup and check the fasttext word embedding file crawl-300d-2M.vec
# Basically, it skips the first line which holds meta information and checks 
# whether the dimensions of the embedding vectors are the same.

import numpy as np

import argparse
import os
import csv


def clean_embedding_file(file_path, result_path, skip_lines):
    i = 0
    with open(file_path) as f, open(result_path, 'w') as out:
        for row in f:
            i = i + 1
            if (i > skip_lines):
                data = row.split(" ")
                word = data[0]
                embedding = np.array([float(num) for num in data[1:-1]])
                if (len(embedding) != 300):
                    print("Record for word " + word + " has size=" + str(len(embedding)))
                else:
                    out.write(row)
 

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess embedding file")

    parser.add_argument("embedding_path")
    parser.add_argument("outfile")
    args = parser.parse_args()

    print("Loading embeddings...")
    clean_embedding_file(args.embedding_path, args.outfile, 1)


if __name__ == "__main__":
    main()
