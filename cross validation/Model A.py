import requests
from nltk import pos_tag


# parameters
remove = True  # removes all the words used in the question from both answers
use_length = False  # If the answer is too short, deduct points

use_cosine = True  # upošteva cosinusno podobnost

openie = 0
# 0 - off
# 1 - on (no coref)
# 2 - on (with coref)
# 3 - Samo za testiranje - Klemen (ker mi ne dela coref)



import string
import nltk
from nltk.stem import WordNetLemmatizer


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

def getTokens(text):
    lowered = text.lower()
    table = text.maketrans({key: None for key in string.punctuation})
    lowered = lowered.translate(table)
    return nltk.word_tokenize(lowered)

def preprocess(text):
    tokens = getTokens(text)
    tagged = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token[0], penn_to_wn(token[1])) for token in tagged]
    return " ".join(lemmas)
    # pos_tag = nltk.pos_tag(lemmas)
    # print(pos_tag)
    # return " ".join([pt[0] for pt in pos_tag if pt[1] == "NN" or pt[1][0:2] == "VB" or pt[1] == "JJ"])


def removeCommmonWords(question, answer, useDumb = False):
    if useDumb:  # bolj primitiven pristop, a deluje bolje v nekaterih primerih
        for word in question.split():
            word = word.replace(".", "")
            word = word.replace(",", "")
            word = word.replace("!", "")
            word = word.replace("?", "")

            answer = answer.replace(" " + word + " ", " ")
            answer = answer.replace(" " + word + ".", "")
            answer = answer.replace(" " + word + ",", "")
            answer = answer.replace(" " + word + "!", "")
        return answer
    else:
        ret = []
        for wordA in getTokens(answer):
            duplicate = False
            for word in getTokens(question):
                if word == wordA:
                    duplicate = True
                    break
            if not duplicate:
                ret.append(wordA)
        return ' '.join([str(x) for x in ret])


def is_long_enough(ref_answer, test_answer):
    len_a = len(getTokens(ref_answer))
    len_b = len(getTokens(test_answer))

    if (float(len_b) / float(len_a)) < 0.5:  # ali je odgovor vsaj polovico toliko dolg kot referenčni?
        return False  # prekratek odgovor
    else:
        return True


def predict():
    f = open("../data/Weightless_dataset_train_A.csv", "r", encoding="utf-8")
    first = True
    questions = []
    answers = []
    grades = []
    texts = []
    for line in f:
        if first:
            first = False
            continue
        s = line.split(";")
        questions.append(s[10])
        answers.append(s[11])
        grades.append(s[14])
        texts.append(s[15])

    f = open("../data/dataset - fixed.csv", "r", encoding="utf8")

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
    # test_answers = [answers_all[i] for i in range(len(questions_all)) if questions_all[i] == questions[0]]
    # test_answers = [answers_all[i] for i in range(len(questions_all))]
    test_answers = []
    test_grades = []
    for q in range(len(questions)):
        test_answers.append([answers_all[i] for i in range(len(questions_all)) if questions_all[i] == questions[q]])
        test_grades.append(
            [float(grades_all[i].replace(",", ".")) for i in range(len(questions_all)) if
             questions_all[i] == questions[q]])
    # test_grades = [float(grades_all[i].replace(",", ".")) for i in range(len(questions_all)) if questions_all[i] == questions[0]]
    # test_grades = [float(grades_all[i].replace(",", ".")) for i in range(len(questions_all))]
    print([len(t) for t in test_answers])

    # vse besede ki so v vprašarnju odstrani iz obeh odgovorov
    if remove:
        for i in range(len(questions)):
            answers[i] = removeCommmonWords(questions[i], answers[i])

        for i in range(len(test_answers)):
            for j in range(len(test_answers[i])):
                test_answers[i][j] = removeCommmonWords(questions[i], test_answers[i][j])


    pre_answer = preprocess(answers[0])
    pre_text = preprocess(texts[0])
    pre_answers = [preprocess(a) for a in answers]
    pre_texts = [preprocess(t) for t in texts]


    #TRIPLES
    BASE_TRIPLES = []
    if openie > 0:  # če je 0, je coref izključen
        for i in range(len(texts)):
            data = answers[i] + ". " + texts[i]
            BASE_TRIPLES.append(openie_extract(data.encode("utf8"), openie))


    from sklearn.feature_extraction.text import TfidfVectorizer


    vect = TfidfVectorizer()  # parameters for tokenization, stopwords can be passed

    documents = []
    for i in range(len(texts)):
        documents.append(pre_answers[i])
        documents.append(pre_texts[i])
    tfidf = vect.fit_transform(documents)
    cosine = (tfidf * tfidf.T).A

    correct = 0

    true_grades = []
    predicted_grades = []

    for q in range(0, len(texts)):  # Loop through all questions
        vect = TfidfVectorizer()  # parameters for tokenization, stopwords can be passed
        # tfidf = vect.fit_transform([texts[0], answers[0]])
        # tfidf = vect.fit_transform([pre_answer, pre_text])
        tfidf = vect.fit_transform([pre_answers[q], pre_texts[q]])
        cosine = (tfidf * tfidf.T).A
        # print("Cosine similarity between the documents: \n{}".format(cosine))
        weights = vect.transform([preprocess(ta) for ta in test_answers[q]])
        predict = tfidf * weights.T
        for i in range(predict.shape[1]):  # Loop through all answers
            p = 0
            p_tfidf = 0
            p_triples = 0
            prediction = max(predict[0, i], predict[1, i])
            if prediction > 0.3:  # 0.5
                p_tfidf = 1
            elif prediction > 0.2:  # 0.3
                p_tfidf = 0.5
            else:
                p_tfidf = 0

            if openie > 0:  # če je 0, potem je coref izključen
                triples = openie_extract(test_answers[q][i].encode("utf8"), openie)
                for bt in BASE_TRIPLES:
                    for t in triples:
                        #if t[0] == bt[0] or t[1] == bt[1] or t[2] == bt[2]:
                        if (t[0] == bt[0] and t[1] == bt[1]) or (t[0] == bt[0] and t[2] == bt[2]) or (
                                t[1] == bt[1] and t[2] == bt[2]):
                            p_triples += 1

                if p_triples >= 1:
                    p_triples = 1
                elif p_triples >= 0.5:
                    p_triples = 0.5
                else:
                    p_triples = 0

                if p_triples == 0:
                    p -= 0.5
                    p = max(p, 0)
                else:
                    p += 0.5
                    p = min(p, 1)

            if use_cosine:  # upošteva kosinusno podobnost
                if openie == 0:  # openie iključen
                    p = p_tfidf
                else:  # upošteva tudi openie
                    # mogoče uporabit drugačno povprečje
                    p_avg = float(p_tfidf + p_triples) / 2
                    if p_avg < 0.25:
                        p = 0
                    elif p_avg < 0.75:
                        p = 0.5
                    else:
                        p = 1
            else:  # upošteva samo openie, brez kosinusne podobnosti
                p = p_triples

            # če je odgovor prekratek, odbije pol točke
            if use_length and not is_long_enough(answers[q], test_answers[q][i]):
                p -= 0.5
                if p < 0:
                    p = 0

            if abs(p - test_grades[q][i]) <= 0:
                correct += 1

            true_grades.append(test_grades[q][i])
            predicted_grades.append(p)
    print("Parametri: remove: ", remove, ", use_length: ", use_length, ", cosine: ", use_cosine, ", openie: ", openie)
    print("Correct: ", correct, "/", len(questions_all))

    tg = [i * 2 for i in true_grades]
    pg = [i * 2 for i in predicted_grades]

    from sklearn.metrics import classification_report

    print(classification_report(tg, pg))


remove = False  # removes all the words used in the question from both answers
use_length = False  # If the answer is too short, deduct points
use_cosine = True  # upošteva cosinusno podobnost
openie = 0
# 0 - off
# 1 - on (no coref)
# 2 - on (with coref)
# 3 - Samo za testiranje - Klemen (ker mi ne dela coref)

predict()
openie = 1
predict()
openie = 2
predict()

openie = 0
remove = True
predict()
openie = 1
predict()
openie = 2
predict()

remove = False
openie = 0
use_length = True
predict()
openie = 1
predict()
openie = 2
predict()

remove = True
openie = 0
predict()
openie = 1
predict()
openie = 2
predict()

remove = False
use_length = False
use_cosine = False
openie = 1
predict()
openie = 2
predict()



'''
TFIDF + openIE
Correct:  126 / 850
C:\ProgramData\Miniconda3\envs\onj\lib\site-packages\sklearn\metrics\classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
              precision    recall  f1-score   support
  'precision', 'predicted', average, warn_for)

         0.0       0.22      0.31      0.26        91
         1.0       0.14      0.66      0.23       148
         2.0       0.00      0.00      0.00       611

   micro avg       0.15      0.15      0.15       850
   macro avg       0.12      0.32      0.16       850
weighted avg       0.05      0.15      0.07       850


'''


'''
samo TFIDF

Correct:  606 / 850
              precision    recall  f1-score   support

         0.0       0.33      0.19      0.24        91
         1.0       0.36      0.18      0.24       148
         2.0       0.78      0.92      0.84       611

   micro avg       0.71      0.71      0.71       850
   macro avg       0.49      0.43      0.44       850
weighted avg       0.66      0.71      0.67       850
'''


'''
TFIDF popravljena lematizacija
Correct:  619 / 850
              precision    recall  f1-score   support

         0.0       0.43      0.16      0.24        91
         1.0       0.36      0.11      0.17       148
         2.0       0.76      0.96      0.85       611

   micro avg       0.73      0.73      0.73       850
   macro avg       0.52      0.41      0.42       850
weighted avg       0.66      0.73      0.67       850
'''