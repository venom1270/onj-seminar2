#!/usr/bin/python

import sys, getopt
import csv
import requests
import json
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

SERVER_URL = "http://localhost:8080/predict"
TEST_FILE = ""
VERBOSE = False


def parseArguments(argv):
    global TEST_FILE
    global SERVER_URL
    global VERBOSE
    VERBOSE = True
    try:
        opts, args = getopt.getopt(argv, "hvt:s:", ["verbose", "test-file=", "server-url="])
    except getopt.GetoptError:
        print('./onj-eval.py -v -t <test-file> [-s <server-url>]')
        sys.exit(2)
    TEST_FILE = "../data/Weightless_dataset_test.csv"
    for opt, arg in opts:
        if opt == '-h':
            print('./onj-eval.py -v -t <test-file> [-s <server-url>]')
            sys.exit()
        elif opt in ("-t", "--test-file"):
            TEST_FILE = arg
        elif opt in ("-s", "--server-url"):
            SERVER_URL = arg
        elif opt in ("-v", "--verbose"):
            VERBOSE = True

    if TEST_FILE == "":
        print("Parameter 'test-file' is mandatory")
        sys.exit(2)


def evaluate():
    for model in ["A", "B", "C"]:
        print("\nEvaluating model %s" % model)

        trueScores = []
        predScores = []
        with open(TEST_FILE, "r", encoding="utf-8") as csvFile:
            reader = csv.DictReader(csvFile)
            for example in reader:
                #print(example)
                trueScore = int(float(example["Final.rating"].replace(",", "."))*10)
                trueScores.append(trueScore)
                #s = example["Question"]
                #print(s)
                req = {
                    "modelId": model,
                    "question": example["Question"],
                    "questionResponse": example["Response"]
                }

                #print()

                res = requests.post(url = SERVER_URL, json = req)
                json_data = json.loads(res.text)
                predScore = int(float(json_data["score"])*10)
                predScores.append(predScore)

        if VERBOSE:
            print("\tTrue scores: %s" % trueScores)
            print("\tPredicted scores: %s\n" % predScores)

        print("\t F1 (macro): %f" % f1_score(trueScores, predScores, average='macro'))
        print("\t F1 (micro): %f" % f1_score(trueScores, predScores, average='micro'))
        print("\t F1 (weighted): %f" % f1_score(trueScores, predScores, average='weighted'))


def main(argv):
    parseArguments(argv)
    evaluate()

if __name__ == "__main__":
    main(sys.argv[1:])