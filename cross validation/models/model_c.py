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
    question = preprocess(question)
    answer = preprocess(answer)
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

DATA_C = read_data()
IMPORTANT_WORDS = []

def train():
    for d in range(len(DATA_C)):
        test_answers = [a[2] for a in DATA_C[d] if a[1] == '1']
        IWords = importantWords(DATA_C[d][0][0], test_answers, threshold=importantWordThreshold, stem=stemWords)
        if numSynonyms > 0:
            IW_syn = []
            for iw in IWords:
                tag = pos_tag([iw])
                IW_syn.append(iw)
                #print(tag)
                syns = getSynonyms(iw, numSynonyms, penn_to_wn(tag[0][1]))
                if syns is not None:
                    print(syns)
                    for i in syns:
                        IW_syn.append(i.replace("_", " "))

            IWords = IW_syn

        IMPORTANT_WORDS.append(IWords)

def predictScore(question, answer):
    questionNumber = -1
    for i in range(len(DATA_C)):
        if DATA_C[i][0][0] == question:
            questionNumber = i
            break
    if questionNumber == -1:
        return None

    vect = TfidfVectorizer()

    answer = removeCommmonWords(question, answer)

    tfidf = vect.fit_transform([" ".join(IMPORTANT_WORDS[i])])
    weights = vect.transform([answer])
    predict = tfidf * weights.T
    predict = predict.toarray()
    #print(predict)

    p = 0

    if predict[0][0] > score10threshold:
        p = 1
    elif predict[0][0] > score05threshold:
        p = 0.5
    else:
        p = 0

    return p


