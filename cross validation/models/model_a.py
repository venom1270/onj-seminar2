import requests
from nltk import pos_tag

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

import string
import nltk
from nltk.stem import WordNetLemmatizer


def openie_extract(text, resolve_coref=True):

    if resolve_coref:
        url = 'http://localhost:9000/?properties={"annotators": "tokenize,ssplit,pos,lemma,openie,coref", "outputFormat": "json", "openie.resolve_coref": "true", "openie.triple.strict": "false", "openie.triple.all_nominals": "false"}'
        # DISABLE COREF # url = 'http://localhost:9000/?properties={"annotators": "tokenize,ssplit,pos,lemma,openie", "outputFormat": "json", "openie.triple.strict": "false", "openie.triple.all_nominals": "false"}'

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


def removeCommmonWords(question, answer):
    for word in question.split():
        answer = answer.replace(word + " ", "")
        answer = answer.replace(word + ".", "")
        answer = answer.replace(word + ",", "")
        answer = answer.replace(word + "!", "")
    return answer


remove = False
if remove:
    for i in range(len(questions)):
        answers[i] = removeCommmonWords(questions[i], answers[i])


pre_answers = [preprocess(a) for a in answers]
pre_texts = [preprocess(t) for t in texts]

trained_base_triples = []

def train():
    for i in range(len(texts)):
        data = answers[i] + ". " + texts[i]
        trained_base_triples.append(openie_extract(data.encode("utf8")))



from sklearn.feature_extraction.text import TfidfVectorizer

'''
vect = TfidfVectorizer()  # parameters for tokenization, stopwords can be passed

documents = []
for i in range(len(texts)):
    documents.append(pre_answers[i])
    documents.append(pre_texts[i])
tfidf = vect.fit_transform(documents)
cosine = (tfidf * tfidf.T).A

'''

def predictScore(question, answer):
    questionNumber = -1
    for i in range(len(questions)):
        if questions[i] == question:
            questionNumber = i
            break
    if questionNumber == -1:
        print(questionNumber)
        return None

    print(questionNumber)

    test_answers = [[answer]]




    correct = 0

    true_grades = []
    predicted_grades = []

# for q in range(0, len(texts)):  # Loop through all questions
    vect = TfidfVectorizer()  # parameters for tokenization, stopwords can be passed
    # tfidf = vect.fit_transform([texts[0], answers[0]])
    # tfidf = vect.fit_transform([pre_answer, pre_text])
    tfidf = vect.fit_transform([pre_answers[questionNumber], pre_texts[questionNumber]])
    cosine = (tfidf * tfidf.T).A
    # print("Cosine similarity between the documents: \n{}".format(cosine))
    weights = vect.transform([preprocess(ta) for ta in test_answers[0]])
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

        triples = openie_extract(test_answers[0][i].encode("utf8"))
        for bt in trained_base_triples:
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

        p = p_tfidf
        return p



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