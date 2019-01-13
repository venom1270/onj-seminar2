import string
import nltk
from nltk.stem import WordNetLemmatizer
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from random import randint

# parameters
use_cosine = True  # upošteva cosinusno podobnost

openie = 2
# 0 - off
# 1 - on (no coref)
# 2 - on (with coref)
# 3 - Samo za testiranje - Klemen (ker mi ne dela coref)

no_of_synonyms = 5  # koliko sinomimov poišče v wordnetu za vsako besedo

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


def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('R'):
        return 'a'

    if tag.startswith('J'):
        return 'r'

    return 'n'

def openie_extract(text, coref_param):
    if coref_param == 1:  # openie, no coref
        url = 'http://localhost:9000/?properties={"annotators": "tokenize,ssplit,pos,lemma,openie,coref", "outputFormat": "json", "openie.resolve_coref": "false", "openie.triple.strict": "false", "openie.triple.all_nominals": "false"}'
    elif coref_param == 2:  # openie with coref
        url = 'http://localhost:9000/?properties={"annotators": "tokenize,ssplit,pos,lemma,openie,coref", "outputFormat": "json", "openie.resolve_coref": "true", "openie.triple.strict": "false", "openie.triple.all_nominals": "false"}'
    elif coref_param == 3:  # samo za testiranje - Klemen
        url = 'http://localhost:9000/?properties={"annotators": "tokenize,ssplit,pos,lemma,openie", "outputFormat": "json", "openie.triple.strict": "false", "openie.triple.all_nominals": "false"}'

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
    tagged = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token[0], penn_to_wn(token[1])) for token in tagged]
    #lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmas)
    # pos_tag = nltk.pos_tag(lemmas)
    # print(pos_tag)
    # return " ".join([pt[0] for pt in pos_tag if pt[1] == "NN" or pt[1][0:2] == "VB" or pt[1] == "JJ"])


def getSynonyms(word, n, partOfSpeech):  # wn.VERB || wn. NOUN || wn.ADJ || wn.ADV
    allSynsets = wn.synsets(word, pos=partOfSpeech)  # Vsi synseti

    if (len(allSynsets) == 0):  # Če ni synsetov ni kaj iskati
        return None
    else:
        allSynsetsUsable = []  # synseti z več kot eno lemo
        for i in range(len(allSynsets)):
            if len(allSynsets[i].lemmas()) > 1:
                allSynsetsUsable.append(allSynsets[i])

        if (len(allSynsetsUsable) == 0):  # Če ni nobenega synseta z vsaj eno sopomenko, ni kaj iskati
            return None
        else:
            ret = []
            for synset in allSynsetsUsable:
                allLemmas = synset.lemmas()  # Vse sopomenke besede v izbranem synsetu
                allLemmasUsable = []  # Drugačne leme

                for i in range(len(allLemmas)):  # Samo drugačne leme
                    if allLemmas[i].name() != word:
                        allLemmasUsable.append(allLemmas[i])

                for lemma in allLemmasUsable:
                    synonym = lemma.name()
                    if synonym not in ret:
                        ret.append(synonym)
                        if len(ret) == n:
                            return ret
            return ret


def formTriples(subject, relation, object, n):
    subject = subject.replace(" ", "_")
    relation = relation.replace(" ", "_")
    object = object.replace(" ", "_")
    synSubjects = getSynonyms(subject, n, wn.NOUN)
    if synSubjects == None:
        synSubjects = [subject]
    synRelations = getSynonyms(relation, n, wn.VERB)
    if synRelations == None:
        synRelations = [relation]
    synObjects = getSynonyms(object, n, None)
    if synObjects == None:
        synObjects = [object]

    ret = []

    for a in synSubjects:
        for b in synRelations:
            for c in synObjects:
                ret.append((a.replace("_", " "), b.replace("_", " "), c.replace("_", " ")))

    return ret

def predict(DATA_train, DATA_test):

    ## DATA[question_number] -> tuples of (queston, grade, answer, text)

    BASE_TRIPLES = []
    if openie > 0:  # če je 0, je coref izključen
        for i in DATA_train:
            data = i[0][3] + ". "  # answers[i] + " " + texts[i]
            for j in i:  # loop through all tuples
                data += j[2] + ". "
            triples = openie_extract(data.encode("utf8"), openie)
            additionalTriples = []
            for triple in triples:
                tripleList = formTriples(triple[0], triple[1], triple[2], no_of_synonyms)
                additionalTriples = additionalTriples + tripleList
            BASE_TRIPLES.append(triples + additionalTriples)

        # print(BASE_TRIPLES)


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
        test_answers = [a[2] for a in DATA_test[d]]
        # if d == 0:
        #    test_answers.append("blabla")
        test_grades = [float(a[1]) for a in DATA_test[d]]

        true_grades += test_grades

    # Cosine similarity with grade 0, 0.5 and 1 training examples

        vect = TfidfVectorizer()  # parameters for tokenization, stopwords can be passed
        # tfidf = vect.fit_transform([pre_answers[d], pre_texts[d]])
        if len(pre_answers_00[d]) != 0:
            tfidf_00 = vect.fit_transform(pre_answers_00[d])
            weights_00 = vect.transform([preprocess(ta) for ta in test_answers])
            predict_00 = tfidf_00 * weights_00.T
        else:
            predict_00 = np.zeros((len(test_answers), len(test_answers)))

        if len(pre_answers_05[d]) != 0:
            tfidf_05 = vect.fit_transform(pre_answers_05[d])
            weights_05 = vect.transform([preprocess(ta) for ta in test_answers])
            predict_05 = tfidf_05 * weights_05.T
        else:
            predict_05 = np.zeros((len(test_answers), len(test_answers)))

        from sklearn.feature_extraction.text import CountVectorizer

        vect = TfidfVectorizer()

        tfidf_10 = vect.fit_transform([pre_texts[d]] + pre_answers_10[d])
        weights_10 = vect.transform([preprocess(ta) for ta in test_answers])
        predict_10 = tfidf_10 * weights_10.T


        # Shiranna feels excited and scared as the shuttle is taking off and it even affects her heart-rate and her temperature.
        # test_answers = ["Excited, scared. It affected her heart rate and temp"]


        for i in range(len(test_answers)):
            all_count += 1  # only for statistics at the end

            test_answer = test_answers[i]
            if openie > 0:  # če je 0, potem je coref izključen
                triples = openie_extract(test_answer.encode("utf8"), openie)

            p = 0
            p_tfidf = 0
            p_triples = 0

            # if len(triples) < 1:
            # prediction = max(predict[0,i], predict[1,i])
            # prediction = max(predict[:,i])
            # prediction = np.argmax([max(predict_00[:,i]), max(predict_10[:,i])])
            prediction = max(predict_10[:, i])

            # print("00: ", max(predict_00[:,i]), "05: ", max(predict_05[:,i]), "10: ", max(predict_10[:,i]))
            # print("---")

            '''
            if prediction == 0:
                p_tfidf = 0
            elif prediction == 1:
                p_tfidf = 1
            '''
            # prediction = max([jaccard_similarity(test_answers[i], base) for base in pre_answers_10[d]])

            # '''
            if prediction > 0.4:  # 0.4
                p_tfidf = 1
            elif prediction > 0.2:  # 0.2
                p_tfidf = 0.5
            else:
                p_tfidf = 0
            # '''
            # else:

            if openie > 0:  # če je 0, potem je coref izključen
                for bt in BASE_TRIPLES[d]:
                    for t in triples:
                        # if t[0] == bt[0] or t[1] == bt[1] or t[2] == bt[2]:
                        if (t[0] == bt[0] and t[1] == bt[1]) or (t[0] == bt[0] and t[2] == bt[2]) or (t[1] == bt[1] and t[2] == bt[2]):
                            # print(t)
                            # print(bt)
                            p_triples += 1
                # p_triples = p_triples*4 / max(len(triples), 1)
                prediction = 0
                if p_triples >= 1:
                    p_triples = 1
                elif p_triples >= 0.5:
                    p_triples = 0.5
                else:
                    p_triples = 0

                if use_cosine:  # Uporabi oboje
                    p = (p_triples + p_tfidf * 2) / 3
                    if p > 0 and p < 1 and p != 0.5:
                        p = 0.5

                    p = p_tfidf
                    if p_triples == 0:
                        p -= 0.5
                        p = max(p, 0)
                    else:
                        p += 0.5
                        p = min(p, 1)
                else:
                    p = p_triples  # Uporabi samo triples
            else:
                p = p_tfidf  # Uporabi samo tfidf

            #p = p_tfidf


            # if d == 0 and i == len(test_answers)-1:
            #    print(p)


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

POGANJAL KLEMEN - BREZ COREF


K =  0.2
1 22 30
Correct:  81 / 166
              precision    recall  f1-score   support

         0.0       0.45      0.26      0.33        34
         1.0       0.17      0.16      0.17        43
         2.0       0.61      0.73      0.67        89

   micro avg       0.49      0.49      0.49       166
   macro avg       0.41      0.39      0.39       166
weighted avg       0.47      0.49      0.47       166

K =  0.4
1 26 29
Correct:  97 / 169
              precision    recall  f1-score   support

         0.0       0.53      0.45      0.49        20
         1.0       0.23      0.23      0.23        39
         2.0       0.71      0.72      0.71       110

   micro avg       0.57      0.57      0.57       169
   macro avg       0.49      0.47      0.48       169
weighted avg       0.57      0.57      0.57       169

K =  0.6000000000000001
2 27 25
Correct:  132 / 170
              precision    recall  f1-score   support

         0.0       0.40      0.20      0.27        10
         1.0       0.24      0.26      0.25        19
         2.0       0.87      0.89      0.88       141

   micro avg       0.78      0.78      0.78       170
   macro avg       0.50      0.45      0.46       170
weighted avg       0.77      0.78      0.77       170

K =  0.8
2 25 29
Correct:  127 / 169
              precision    recall  f1-score   support

         0.0       0.78      0.47      0.58        15
         1.0       0.22      0.32      0.26        19
         2.0       0.86      0.84      0.85       135

   micro avg       0.75      0.75      0.75       169
   macro avg       0.62      0.54      0.56       169
weighted avg       0.78      0.75      0.76       169

K =  1.0
2 28 27
Correct:  121 / 176
              precision    recall  f1-score   support

         0.0       0.40      0.50      0.44        12
         1.0       0.26      0.32      0.29        28
         2.0       0.83      0.78      0.81       136

   micro avg       0.69      0.69      0.69       176
   macro avg       0.50      0.53      0.51       176
weighted avg       0.71      0.69      0.70       176

F1 micro:  0.6554732364892917
F1 macro:  0.4816253659912779

Process finished with exit code 0



'''