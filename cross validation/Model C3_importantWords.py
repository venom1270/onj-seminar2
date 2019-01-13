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


# parameters
use_cosine = True  # upošteva cosinusno podobnost

openie = 0
# 0 - off
# 1 - on (no coref)
# 2 - on (with coref)
# 3 - Samo za testiranje - Klemen (ker mi ne dela coref)

#Presledek spredaj je pomemben!
additionalWordsToRemoveFromQuestion = " and or she he it his her its"

numSynonyms = 0
stemWords = False
importantWordThreshold = 0.02

score10threshold = 0.5
score05threshold = 0.4

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

def removeWord(word, text):
    res = []
    for i in getTokens(text):
        if i != word:
            res.append(i)
    return " ".join(res)

def stemText(text):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(t) for t in getTokens(text)])

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

def importantWords(question, answers, stem=False, threshold=0):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()


    question = preprocess(question)
    answers_1 = [removeCommmonWords(question + additionalWordsToRemoveFromQuestion, preprocess(a)) for a in answers]

    answers = []
    if stem:
        for a in answers_1:
            #qwe = [stemmer.stem(q) for q in getTokens(a)]
            answers.append(stemText(a))
        #answers = [" ".join(stemmer.stem(b)) for b in getTokens(a) for a in answers_1]

    else:
        answers = answers_1
    #print(answers)

    important_words = []

    for i in range(len(answers)):
        ans_test = answers[i]

        ans_train = answers[:i] + answers[i+1:]
        vect = TfidfVectorizer()  # parameters for tokenization, stopwords can be passed
        tfidf = vect.fit_transform(ans_train)
        weights = vect.transform([ans_test])
        predict = tfidf * weights.T
        baseline = sum(predict[:, 0]) / len(ans_train)

        for word in getTokens(ans_test):
            #print("Removing word: ", word)
            test = removeWord(word, ans_test)
            weights = vect.transform([test])
            predict = tfidf * weights.T
            #print(predict)
            score = sum(predict[:,0])/len(ans_train)
            #print("New text: ", test)
            #print(score)
            if score.toarray()+threshold < baseline:
                if word not in important_words:
                    important_words.append(word)

    #print("IMPORTANT WORDS: ", important_words)
    return important_words



def predict(DATA_train, DATA_test):

    ## DATA[question_number] -> tuples of (queston, grade, answer, text)


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

        IMPORTANT_WORDS = importantWords(DATA_test[d][0][0], test_answers, threshold=importantWordThreshold, stem=stemWords)

        if numSynonyms > 0:
            IW_syn = []
            for iw in IMPORTANT_WORDS:
                tag = pos_tag([iw])
                IW_syn.append(iw)
                #print(tag)
                syns = getSynonyms(iw, numSynonyms, penn_to_wn(tag[0][1]))
                if syns is not None:
                    print(syns)
                    for i in syns:
                        IW_syn.append(i.replace("_", " "))

            IMPORTANT_WORDS = IW_syn

        #print(IMPORTANT_WORDS)

        #test_answers = [stemText(ta) for ta in test_answers]


        #continue

        vect = TfidfVectorizer()

        tfidf = vect.fit_transform([" ".join(IMPORTANT_WORDS)])
        weights = vect.transform([preprocess(ta) for ta in test_answers])
        predict = tfidf * weights.T
        predict = predict.toarray()
        #print(predict)

        for i in range(len(test_answers)):

            all_count += 1

            p = 0

            if predict[0][i] > score10threshold:
                p = 1
            elif predict[0][i] > score05threshold:
                p = 0.5
            else:
                p = 0

            predicted_grades.append(p)
            #print("Predicted:", p, " Real:", test_grades[i])


            if p == test_grades[i]:
                correct += 1

    print("Correct: ", correct, "/", all_count)

    return true_grades, predicted_grades, correct, all_count


#q = "How does Shiranna feel as the shuttle is taking off?"
#a = ["Shiranna feels nervous and excited at the same time.", "Nervous and a liitle bit excited.", "Nervous and excited because the shuffle is taking off."]
#q = "How does Shiranna feel as the shuttle is taking off?"
#a = ["Nervous, but also excited to be with her mother.", "Shiranna feels both excited and nervous as the shuttle is taking off.", "Shiranna feels excited and scared as the shuttle is taking off and it even affects her heart-rate and her temperature."]

#importantWords(q, a)

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

print("MICRO: ", F1_micro)
print("MACRO: ", F1_macro)

'''
if threshodls 0.5, 0.4
threshold = 0.02 , stem=True

C:\ProgramData\Miniconda3\envs\onj\python.exe "C:/Users/zigsi/Desktop/FRI git/onj-seminar2/cross validation/Model C3_importantWords.py"
K =  0.2
1 22 30
Correct:  94 / 166
              precision    recall  f1-score   support

         0.0       0.57      0.24      0.33        34
         1.0       0.42      0.26      0.32        43
         2.0       0.60      0.84      0.70        89

   micro avg       0.57      0.57      0.57       166
   macro avg       0.53      0.44      0.45       166
weighted avg       0.55      0.57      0.52       166

K =  0.4
1 26 29
Correct:  104 / 169
              precision    recall  f1-score   support

         0.0       0.43      0.45      0.44        20
         1.0       0.25      0.13      0.17        39
         2.0       0.70      0.82      0.76       110

   micro avg       0.62      0.62      0.62       169
   macro avg       0.46      0.47      0.45       169
weighted avg       0.57      0.62      0.58       169

K =  0.6000000000000001
2 27 25
Correct:  141 / 170
              precision    recall  f1-score   support

         0.0       0.60      0.60      0.60        10
         1.0       0.35      0.37      0.36        19
         2.0       0.91      0.91      0.91       141

   micro avg       0.83      0.83      0.83       170
   macro avg       0.62      0.63      0.62       170
weighted avg       0.83      0.83      0.83       170

K =  0.8
2 25 29
Correct:  130 / 169
              precision    recall  f1-score   support

         0.0       0.23      0.20      0.21        15
         1.0       0.33      0.21      0.26        19
         2.0       0.85      0.91      0.88       135

   micro avg       0.77      0.77      0.77       169
   macro avg       0.47      0.44      0.45       169
weighted avg       0.74      0.77      0.75       169

K =  1.0
2 28 27
Correct:  121 / 176
              precision    recall  f1-score   support

         0.0       0.31      0.42      0.36        12
         1.0       0.29      0.43      0.34        28
         2.0       0.88      0.76      0.82       136

   micro avg       0.69      0.69      0.69       176
   macro avg       0.49      0.54      0.51       176
weighted avg       0.75      0.69      0.71       176

MICRO:  0.6935584419124462
MACRO:  0.4971760969390549

Process finished with exit code 0


'''


'''
if threshodls 0.5, 0.4
threshold = 0.02 , stem=FALSE

C:\ProgramData\Miniconda3\envs\onj\python.exe "C:/Users/zigsi/Desktop/FRI git/onj-seminar2/cross validation/Model C3_importantWords.py"
K =  0.2
1 22 30
Correct:  90 / 166
              precision    recall  f1-score   support

         0.0       0.55      0.18      0.27        34
         1.0       0.39      0.21      0.27        43
         2.0       0.57      0.84      0.68        89

   micro avg       0.54      0.54      0.54       166
   macro avg       0.50      0.41      0.41       166
weighted avg       0.52      0.54      0.49       166

K =  0.4
1 26 29
Correct:  109 / 169
              precision    recall  f1-score   support

         0.0       0.50      0.50      0.50        20
         1.0       0.23      0.08      0.12        39
         2.0       0.71      0.87      0.78       110

   micro avg       0.64      0.64      0.64       169
   macro avg       0.48      0.48      0.47       169
weighted avg       0.57      0.64      0.59       169

K =  0.6000000000000001
2 27 25
Correct:  142 / 170
              precision    recall  f1-score   support

         0.0       0.50      0.60      0.55        10
         1.0       0.31      0.21      0.25        19
         2.0       0.91      0.94      0.92       141

   micro avg       0.84      0.84      0.84       170
   macro avg       0.57      0.58      0.57       170
weighted avg       0.82      0.84      0.83       170

K =  0.8
2 25 29
Correct:  135 / 169
              precision    recall  f1-score   support

         0.0       0.50      0.20      0.29        15
         1.0       0.46      0.32      0.37        19
         2.0       0.84      0.93      0.88       135

   micro avg       0.80      0.80      0.80       169
   macro avg       0.60      0.48      0.51       169
weighted avg       0.77      0.80      0.77       169

K =  1.0
2 28 27
Correct:  132 / 176
              precision    recall  f1-score   support

         0.0       0.18      0.17      0.17        12
         1.0       0.38      0.39      0.39        28
         2.0       0.88      0.88      0.88       136

   micro avg       0.75      0.75      0.75       176
   macro avg       0.48      0.48      0.48       176
weighted avg       0.75      0.75      0.75       176

MICRO:  0.714249954918875
MACRO:  0.48748890851008786

Process finished with exit code 0


'''