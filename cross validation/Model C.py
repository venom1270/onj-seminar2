import string
import nltk
from nltk.stem import WordNetLemmatizer
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from nltk.corpus import wordnet
from nltk import pos_tag
import math

# INTERESTING: https://nlpforhackers.io/wordnet-sentence-similarity/

def read_data():
    f = open("../data/dataset - fixed.csv", "r", encoding="utf-8")

    first = True
    questions_all = []
    answers_all = []
    grades_all = []
    texts_all = []
    for line in f:
        if first:
            first = False
            continue
        s = line.split(";")
        questions_all.append(s[10])
        answers_all.append(s[11])
        grades_all.append(s[14])
        texts_all.append(s[15])
        if s[15] == '1':
            print(line)

    ## DATA[question_number] -> tuples of (queston, grade, answer, text)
    old = questions_all[0]
    DATA = []
    data_tmp = []
    for i in range(len(questions_all)):
        if questions_all[i] == old:
            triple = (questions_all[i], grades_all[i], answers_all[i], texts_all[i])
            data_tmp.append(triple)
        else:
            old = questions_all[i]
            DATA.append(data_tmp)
            data_tmp = []
            data_tmp.append((questions_all[i], grades_all[i], answers_all[i], texts_all[i]))
    DATA.append(data_tmp)

    #print(len(DATA))
    #for i in DATA:
    #    print(len(i))

    return DATA


def openie_extract(text, resolve_coref=True):

    if resolve_coref:
        url = 'http://localhost:9000/?properties={"annotators": "tokenize,ssplit,pos,lemma,openie,coref", "outputFormat": "json", "openie.resolve_coref": "true", "openie.triple.strict": "false", "openie.triple.all_nominals": "false"}'
    else:
        url = 'http://localhost:9000/?properties={"annotators": "tokenize,ssplit,pos,lemma,openie,coref", "outputFormat": "json", "openie.resolve_coref": "false", "openie.triple.strict": "false", "openie.triple.all_nominals": "false"}'
    data = text
    response = requests.post(url, data=data)
    response.encoding = "utf-8"
    triples = []
    for s in response.json()["sentences"]:
        for i in s["openie"]:
            # print("Subject:", i["subject"], " | Relation:", i["relation"], " | Object:", i["object"])
            # triples.append((lemmatize([i["subject"]])[0], lemmatize([i["relation"]])[0], lemmatize([i["object"]])[0]))
            triples.append((i["subject"], i["relation"], i["object"]))
    return triples


def getTokens(text):
    lowered = text.lower()
    table = text.maketrans({key: None for key in string.punctuation})
    lowered = lowered.translate(table)
    return nltk.word_tokenize(lowered)



def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas



def preprocess(text):
    tokens = getTokens(text)
    lemmatizer = WordNetLemmatizer()
    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(token[0], penn_to_wn(token[1], default='n')) for token in tagged]
    #lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmas)
    # pos_tag = nltk.pos_tag(lemmas)
    # print(pos_tag)
    # return " ".join([pt[0] for pt in pos_tag if pt[1] == "NN" or pt[1][0:2] == "VB" or pt[1] == "JJ"])


def penn_to_wn(tag, default=None):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return default


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wordnet.synsets(word, wn_tag)[0]
    except:
        return None

def predict(DATA_train, DATA_test):

    ## DATA[question_number] -> tuples of (queston, grade, answer, text)

    '''

    BASE_TRIPLES = []
    for i in DATA_train:
        data = i[0][3] + " "  # answers[i] + " " + texts[i]
        for j in i:  # loop through all tuples
            data += j[2] + " "
        BASE_TRIPLES.append(openie_extract(data.encode("utf8")))
    '''


    pre_answers_00 = []
    pre_answers_05 = []
    pre_answers_10 = []
    pre_texts = []
    for i in DATA_train:
        pre_answers_00.append([preprocess(ans[2]) for ans in i if ans[1] == '0' and len(ans[2].split(" ")) > 3])
        pre_answers_05.append([preprocess(ans[2]) for ans in i if ans[1] == '0.5' and len(ans[2].split(" ")) > 2])
        pre_answers_10.append([preprocess(ans[2]) for ans in i if ans[1] == '1' and len(ans[2].split(" ")) > 0])
        pre_texts.append(preprocess(i[0][3]))
    print(len(pre_answers_00[0]), len(pre_answers_05[0]), len(pre_answers_10[0]))


    # weights = vect.transform(test_answers)

    correct = 0

    all_count = 0

    # for F1 scoring
    true_grades = []
    predicted_grades = []

    ## DATA[question_number] -> triples of (queston, grade, answer, text)
    for d in range(len(DATA_test)):
        print("D =", d)
        test_answers = [a[2] for a in DATA_test[d]]
        # if d == 0:
        #    test_answers.append("blabla")
        test_grades = [float(a[1]) for a in DATA_test[d]]

        true_grades += test_grades

        for i in range(len(test_answers)):
            all_count += 1  # only for statistics at the end

            test_answer = test_answers[i]

            tokens = getTokens(test_answer)
            tagged = pos_tag(tokens)

            max_score = 0


            synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in tagged]
            synsets1 = [ss for ss in synsets1 if ss]

            for a in pre_answers_10[d]:
                tagged_base = pos_tag(getTokens(a))
                # Get the synsets for the tagged words

                synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in tagged_base]

                # Filter out the Nones

                synsets2 = [ss for ss in synsets2 if ss]

                synset1_score, count = 0.0, 0

                # For each word in the first sentence

                synset1_idf_sum = 0

                for synset in synsets1:
                    # Get the similarity value of the most similar word in the other sentence
                    #print([ss for ss in synsets2])
                    arr = [synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss)]
                    if len(arr) == 0:
                        continue
                    best_score = max(arr)
                    #best_score = 0
                    #for ss in synsets2:
                    #    best_score = max(score, synset.path_similarity(ss))

                    idf_bottom = len([1 for t in pre_answers_10[d] if synset.lemmas()[0].name() in t])
                    idf = 0
                    idf = math.log(len(pre_answers_10[d]) / (1+idf_bottom))
                    best_score *= idf
                    synset1_idf_sum += idf

                    # Check that the similarity could have been computed
                    if best_score is not None:
                        synset1_score += best_score
                        count += 1

                #if count != 0:
                #    score /= count
                if synset1_idf_sum != 0:
                    synset1_score = synset1_score / synset1_idf_sum
                else:
                    synset1_score = 0
                #print(score)

                synset2_score = 0
                synset2_idf_sum = 0
                for synset in synsets2:
                    arr = [synset.path_similarity(ss) for ss in synsets1 if synset.path_similarity(ss)]
                    if len(arr) == 0:
                        continue
                    best_score = max(arr)
                    idf_bottom = len([1 for t in pre_answers_10[d] if synset.lemmas()[0].name() in t])
                    idf = 0

                    idf = math.log(len(pre_answers_10[d]) / (1+idf_bottom))
                    best_score *= idf
                    synset2_idf_sum += idf

                    # Check that the similarity could have been computed
                    if best_score is not None:
                        synset2_score += best_score

                if synset2_idf_sum != 0:
                    synset2_score = synset2_score / synset2_idf_sum
                else:
                    synset2_score = 0

                final_score = (synset1_score + synset2_score)/2

                max_score = max(max_score, final_score)

            if max_score > 0.6:
                p = 1
            elif max_score > 0.4:
                p = 0.5
            else:
                p = 0

            predicted_grades.append(p)
            # print("Predicted:", p, " Real:", test_grades[i], " --- ", len(triples))


            if p == test_grades[i]:
                correct += 1

    print("Correct: ", correct, "/", all_count)

    return true_grades, predicted_grades, correct, all_count


DATA = read_data()




ratio = 0.8  # 0.8 train data, 0.2 test data FOR EACH QUESTION

F1_micro = 0
F1_macro = 0

k = 0.2
while k <= 1:
    print("K = ", k)

    DATA_train = []
    DATA_test = []
    for i in DATA:
        split1 = int(len(i) * (k-0.2))
        split2 = int(len(i) * k)
        DATA_train.append(i[0:split1] + i[split2:])  # if good_ans[1] == '1'])
        DATA_test.append(i[split1:split2])
    k += 0.2

    true_grades, predicted_grades, correct, all_count = predict(DATA_train, DATA_test)

    tg = [i * 2 for i in true_grades]
    pg = [i * 2 for i in predicted_grades]

    print(classification_report(tg, pg))

    F1_micro += f1_score(tg, pg, average="micro")
    F1_macro += f1_score(tg, pg, average="macro")


F1_micro /= 5.0
F1_macro /= 5.0

print("F1 micro: ", F1_micro)
print("F1 macro: ", F1_macro)

'''
C:\ProgramData\Miniconda3\envs\onj\python.exe "C:/Users/zigsi/Desktop/FRI git/onj-seminar2/cross validation/Model C.py"
K =  0.2
1 22 30
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  88 / 166
              precision    recall  f1-score   support

         0.0       0.25      0.03      0.05        34
         1.0       0.24      0.09      0.13        43
         2.0       0.57      0.93      0.71        89

   micro avg       0.53      0.53      0.53       166
   macro avg       0.35      0.35      0.30       166
weighted avg       0.42      0.53      0.43       166

K =  0.4
1 26 29
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  108 / 169
              precision    recall  f1-score   support

         0.0       0.67      0.30      0.41        20
         1.0       0.18      0.05      0.08        39
         2.0       0.67      0.91      0.77       110

   micro avg       0.64      0.64      0.64       169
   macro avg       0.51      0.42      0.42       169
weighted avg       0.56      0.64      0.57       169

K =  0.6000000000000001
2 27 25
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  132 / 170
              precision    recall  f1-score   support

         0.0       0.50      0.10      0.17        10
         1.0       0.00      0.00      0.00        19
         2.0       0.82      0.93      0.87       141

   micro avg       0.78      0.78      0.78       170
   macro avg       0.44      0.34      0.35       170
weighted avg       0.71      0.78      0.73       170

K =  0.8
2 25 29
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  130 / 169
              precision    recall  f1-score   support

         0.0       0.43      0.20      0.27        15
         1.0       0.00      0.00      0.00        19
         2.0       0.84      0.94      0.89       135

   micro avg       0.77      0.77      0.77       169
   macro avg       0.42      0.38      0.39       169
weighted avg       0.71      0.77      0.73       169

K =  1.0
2 28 27
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  133 / 176
              precision    recall  f1-score   support

         0.0       0.62      0.42      0.50        12
         1.0       0.00      0.00      0.00        28
         2.0       0.80      0.94      0.86       136

   micro avg       0.76      0.76      0.76       176
   macro avg       0.47      0.45      0.45       176
weighted avg       0.66      0.76      0.70       176

F1 micro:  0.6941113824026924
F1 macro:  0.3814038702409588

Process finished with exit code 0


'''


# RUN 2 (with tfidf_bottom + 1)

'''
C:\ProgramData\Miniconda3\envs\onj\python.exe "C:/Users/zigsi/Desktop/FRI git/onj-seminar2/cross validation/Model C.py"
K =  0.2
1 22 30
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  93 / 166
              precision    recall  f1-score   support

         0.0       0.67      0.18      0.28        34
         1.0       0.32      0.28      0.30        43
         2.0       0.62      0.84      0.72        89

   micro avg       0.56      0.56      0.56       166
   macro avg       0.54      0.43      0.43       166
weighted avg       0.56      0.56      0.52       166

K =  0.4
1 26 29
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  98 / 169
              precision    recall  f1-score   support

         0.0       0.55      0.30      0.39        20
         1.0       0.17      0.13      0.14        39
         2.0       0.68      0.79      0.73       110

   micro avg       0.58      0.58      0.58       169
   macro avg       0.46      0.41      0.42       169
weighted avg       0.55      0.58      0.56       169

K =  0.6000000000000001
2 27 25
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  124 / 170
              precision    recall  f1-score   support

         0.0       0.40      0.20      0.27        10
         1.0       0.13      0.21      0.16        19
         2.0       0.87      0.84      0.86       141

   micro avg       0.73      0.73      0.73       170
   macro avg       0.47      0.42      0.43       170
weighted avg       0.76      0.73      0.74       170

K =  0.8
2 25 29
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  113 / 169
              precision    recall  f1-score   support

         0.0       0.86      0.40      0.55        15
         1.0       0.03      0.05      0.04        19
         2.0       0.83      0.79      0.81       135

   micro avg       0.67      0.67      0.67       169
   macro avg       0.57      0.41      0.46       169
weighted avg       0.75      0.67      0.70       169

K =  1.0
2 28 27
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  118 / 176
              precision    recall  f1-score   support

         0.0       0.57      0.33      0.42        12
         1.0       0.16      0.21      0.18        28
         2.0       0.82      0.79      0.81       136

   micro avg       0.67      0.67      0.67       176
   macro avg       0.52      0.45      0.47       176
weighted avg       0.70      0.67      0.68       176

F1 micro:  0.6417255968150042
F1 macro:  0.44321229026431785

Process finished with exit code 0
'''


#RUN 3 (added proper lemmatization)

'''
C:\ProgramData\Miniconda3\envs\onj\python.exe "C:/Users/zigsi/Desktop/FRI git/onj-seminar2/cross validation/Model C.py"
K =  0.2
1 22 30
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  98 / 166
              precision    recall  f1-score   support

         0.0       0.89      0.24      0.37        34
         1.0       0.38      0.35      0.36        43
         2.0       0.64      0.84      0.73        89

   micro avg       0.59      0.59      0.59       166
   macro avg       0.63      0.48      0.49       166
weighted avg       0.62      0.59      0.56       166

K =  0.4
1 26 29
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  99 / 169
              precision    recall  f1-score   support

         0.0       0.60      0.30      0.40        20
         1.0       0.20      0.15      0.17        39
         2.0       0.67      0.79      0.73       110

   micro avg       0.59      0.59      0.59       169
   macro avg       0.49      0.41      0.43       169
weighted avg       0.56      0.59      0.56       169

K =  0.6000000000000001
2 27 25
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  121 / 170
              precision    recall  f1-score   support

         0.0       0.25      0.10      0.14        10
         1.0       0.12      0.21      0.15        19
         2.0       0.88      0.82      0.85       141

   micro avg       0.71      0.71      0.71       170
   macro avg       0.42      0.38      0.38       170
weighted avg       0.76      0.71      0.73       170

K =  0.8
2 25 29
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  117 / 169
              precision    recall  f1-score   support

         0.0       1.00      0.47      0.64        15
         1.0       0.08      0.16      0.11        19
         2.0       0.86      0.79      0.82       135

   micro avg       0.69      0.69      0.69       169
   macro avg       0.65      0.47      0.52       169
weighted avg       0.78      0.69      0.73       169

K =  1.0
2 28 27
D = 0
D = 1
D = 2
D = 3
D = 4
D = 5
D = 6
D = 7
D = 8
D = 9
D = 10
D = 11
Correct:  118 / 176
              precision    recall  f1-score   support

         0.0       0.57      0.33      0.42        12
         1.0       0.15      0.18      0.16        28
         2.0       0.81      0.80      0.80       136

   micro avg       0.67      0.67      0.67       176
   macro avg       0.51      0.44      0.46       176
weighted avg       0.69      0.67      0.68       176

F1 micro:  0.6501374411991541
F1 macro:  0.4573741644266425

Process finished with exit code 0
'''