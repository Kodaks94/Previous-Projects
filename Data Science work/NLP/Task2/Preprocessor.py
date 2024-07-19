import re

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

'''
    CONSTANTS
--------------------------------------------------------------------------------------------------------------
'''
Sentiment_lexiconFile = "./SentimentLexicons/subjclueslen1-HLTEMNLP05.tff"
TRAIN_FILE = "train.tsv"
TEST_FILE = "test.tsv"
#use Debugging mode to either check all the methods accuracy (TRUE), write in the test submission csv file (FALSE)
DEBUGGING_MODE = False
SUBMISSION_FILE = "sub1.csv"
Negation_File = "Negation_words.txt"
nltk.download("stopwords")
ALL_THE_STOPWORDS = [Word for Word in stopwords.words('english') if Word not in ['t', 'not', 'don', 'can', 'no']]
'''
---------------------------------------------------------------------------------------------------------------
'''
import os


"""
Utility functions

1- convert_dataset : the method will allow the program to load either test file or train files
2- write_to_csv : this method was not used  but it is an easy way to keep the processed words 
3- Classifier_process: uses the naive bayes classifier to classify data 

"""

def convert_dataset(filename, isTrainData, IsNumpy):
    """
    :param filename: file name to open
    :param isTrainData: is the file the training data
    :param IsNumpy: do you want to return the array as numpy array
    :return: the array of features
    """
    try:
        dataset = pd.read_csv(filename, sep='\t', header=0)
        dataset.dropna(axis='index', how='any')
        if (isTrainData):
            arrayof_Features = dataset.drop(['PhraseId','SentenceId'], axis=1)
            arrayof_Targets = dataset.drop(['PhraseId', 'SentenceId', 'Phrase'], axis=1)
            if (IsNumpy):

                return arrayof_Features.values, np.transpose(arrayof_Targets.values)[0]
            else:

                return arrayof_Features, arrayof_Targets
        else:
            if (IsNumpy):
                arrayof_features =dataset.drop(['SentenceId'],axis = 1)
                return arrayof_features.values
            else:
                return dataset
    except:
        os.chdir(filename)
        convert_dataset(filename, isTrainData, IsNumpy)

def write_to_csv(sets_of_features, options):
    """

    :param sets_of_features: set of features to write
    :param options: option for what process method file to save to
    :return:
    """
    path = ""
    if options == 1:
        path = "preprocessedPhase.csv"
    elif options == 2:
        path = "SL_Phase.csv"
    elif options == 3:
        path = "NOT_Phase.csv"
    elif options == 4:
        path = "Bigram_Phase.csv"
    file = open(path, 'w')
    featurenames = sets_of_features[0][0].keys()
    name_lines = replace_special_letters(featurenames)
    file.write(name_lines)
    file.write('\n')

    replace_sentiment(feature_sets=sets_of_features, feature_names=featurenames,file=file)

    file.close()

def Classifier_process(train_features, test_features,Classifier):
    """
    either has the debugging mode which only used to see all the accuracy of the files + confusion matrix + analytics
    :param train_features:
    :param test_features:
    :param Classifier: passing the classifier if needed for incremental training
    :return:
    """
    if(DEBUGGING_MODE):
        test = []

        if (test_features == None):

            split_size = int(len(train_features) * 0.1)

            test = train_features[:split_size]

        else:
            test = test_features



        print("Size of the train data", len(train_features))
        print("Size of the test data", len(test))

        Classifier = Classifier.train(train_features)


        print("Accuracy without test_features: ")
        print(nltk.classify.accuracy(Classifier, test))
        print("... most informative feature")
        print(Classifier.show_most_informative_features(30))
        '''
        Confusion matrix
        '''
        refernce = []
        tested = []
        for (x, y) in test:
            refernce.append(y)
            tested.append(Classifier.classify(x))
        print("\n")
        print("Confusion matrix : ///--")
        print(nltk.ConfusionMatrix(refernce, tested))
        return Classifier
    else:

        Classifier = Classifier.train(train_features)


        return Classifier





"""
----------------------------------------------------------------------------
"""

"""
Sentiment Lexicon Process
-------------------------
1- polarity_check
2- get_lexiconFile
3- Score_system
4- lexicon_splitter
5- preprocessing_SLFeatures
"""

def polarity_check(mylist):
    """
    checks the polarity and returns the list
    :param mylist:
    :return:
    """
    if 'positive_count' not in mylist:
        mylist['positive_count'] = 0
    if 'negative_count' not in mylist:
        mylist['negative_count'] = 0
    return mylist

def get_lexiconFile(path):
    """
    loads the sentiment lexicon file
    :param path:
    :return:
    """

    try:
        f = open(path,'r')

        temp = {}
        for i in f:
            no_white_space = i.split()
            word, s_score, p_tag, stemmed, pol = lexicon_splitter(no_white_space)
            temp[word] = [s_score,p_tag,stemmed,pol]
        return temp
    except:
        os.chdir(path)
        get_lexiconFile(path)

def Score_system(mydict,s_score, pol):

    if pol == 'positive':

        if s_score == 'weaksubj':
            mydict["WeakPositive"] += 1
        elif s_score == 'strongsubj':
            mydict["StrongPositive"] += 1
    if pol == 'negative':

        if s_score == 'weaksubj':
            mydict["WeakNegative"] += 1
        elif s_score == 'strongsubj':
            mydict["StrongNegative"] += 1
    return mydict

def lexicon_splitter(line):

    s_score = line[0].split("=")[1]
    stem = line[4].split("=")[1]
    p_tag = line[3].split("=")[1]
    pol = line[5].split("=")[1]
    word = line[2].split("=")[1]
    if (stem != 'y'):
        stemmed = False
    else:
        stemmed = True
    return word,s_score,p_tag,stemmed,pol

def preprocessing_SLFeatures(features_list, doc,sentimentLexicon):

    normalised_feature_list = normal_features(doc, features_list,None,None)

    word_credit = {
        "WeakPositive": 0,
        "StrongPositive": 0,
        "StrongNegative": 0,
        "WeakNegative": 0
    }

    for word in features_list:

        if word in sentimentLexicon:
            s_score, p_tag, stemmed, pol = sentimentLexicon[word]
            word_credit = Score_system(word_credit, s_score,pol)

            normalised_feature_list['negative_count'] = int(word_credit["WeakNegative"]) + (
                    2 * int(word_credit["StrongNegative"]))
            normalised_feature_list['positive_count'] = int(word_credit["WeakPositive"]) + (
                    2 * int(word_credit["StrongPositive"]))

        normalised_feature_list = polarity_check(mylist=normalised_feature_list)


    return normalised_feature_list

"""
-----------------------------------------------------------------------------
"""

"""
Normal Pre-Process
-------------------
1- replace_special_letters
2- replace_sentiment
3- sentiment_switchCase
4- get_word_utility
5- normal_features
6- process_to_output_trainData
7- Create_featuresets
8- preprocessing
9- Word_caseProcess
"""
def replace_special_letters(featureValues):

    nameline = ''
    for i in featureValues:
        i = i.replace(',', 'CN')
        i = i.replace('"', 'QE')
        i = i.replace("'", "SQ")
        nameline += i +','
        nameline +='class'

    return nameline

def replace_sentiment(feature_sets, feature_names,file):

    for feature in feature_sets:

        line = ''
        for key in feature_names:
            line += str(feature[0][key])
            line += ','
        line += sentiment_switchCase(feature[1])
        file.write(line)
        file.write('\n')

def sentiment_switchCase(sentiment):

    if sentiment == 0:
        return str("negative")
    elif sentiment == 1:
        return str("somewhat negative")
    elif sentiment == 2:
        return str("neutral")
    elif sentiment == 3:
        return str("somewhat positive")
    elif sentiment == 4:
        return str("positive")

def get_word_utility(docs, option):
    '''
        Options:
        1 - get words
        2- get features
    '''

    output = []
    if option == 1:
        for(w, s) in docs:
            words = [x for x in w if len(x) >= 3]
            output.extend(words)
        return output

    elif option == 2:
        words = nltk.FreqDist(docs)
        output = [x for (x,y) in words.most_common(200)]
        return output

def normal_features(document, word_features, bigrams,option):
    '''
    Options
    1 = normal
    2 = for Negation features
    3 = Bigram feature
    :param document:
    :param word_features:
    :return:
    '''


    features = {}
    for word in word_features:
        if option != None:
            if option == 2:
                features['Holds({})'.format(word)] = False
                features['Holds(NOT{})'.format(word)] = False
        else:
            features['Holds({})'.format(word)] = (word in set(document))
    if option != None:
        if option == 3:
            for b in bigrams:
                existance = b in nltk.bigrams(document)
                features['BGRAM({}{})'.format(b[0], b[1])] = existance

    return features

def Create_featuresets(feature_list, processed_features,tokens, option):
    '''
    :param option:
    if option == 1 : return for normal preprocessed
    if option == 2 : return for sentiment lexicon features
    if option == 3: return for negation features
    if option ==4 : return for Bigram features
    :return:
    '''
    if option == 1:
        #chunks = [(feature_list[:int(len(feature_list)/4)],processed_features[:int(len(processed_features)/4)]),(feature_list[int(len(feature_list)/4):int(2*len(feature_list)/4)],processed_features[int(len(processed_features)/4):int(2*len(processed_features)/4)]),(feature_list[int(2*len(feature_list)/4):int(3*len(feature_list)/4)],processed_features[int(2*len(processed_features)/4):int(3*len(processed_features)/4)]),(feature_list[int(3*len(feature_list)/4):int(len(feature_list))],processed_features[int(3*len(processed_features)/4):len(processed_features)])]
        #normal = []
        #for fchunk, pchunk in chunks:

            #normal.append([(normal_features(i, fchunk, None, None), j) for (i, j) in pchunk])



        #print(normal)
        return [(normal_features(i, feature_list, None, None), j) for (i, j) in processed_features]
    elif option == 2:
        sentimentLexicon = get_lexiconFile(Sentiment_lexiconFile)
        return [(preprocessing_SLFeatures(i, feature_list, sentimentLexicon),j) for (i,j) in processed_features]
    elif option == 3:
        negation_words = Load_negaton_words(Negation_File)
        return [(Neg_preprocess(i, feature_list,negation_words),j) for (i,j) in processed_features]
    elif option == 4:
        b_features = BIGRAM_features(tokens)
        return [(BIGRAM_process(i, feature_list, b_features), j) for (i, j) in processed_features]



def preprocessing(arrayof_Features, option,Classifier):
    """

    :param arrayof_Features:
    :param option: to choose which method to apply preprocessing
    :param Classifier: passing the classifier for incremental training
    :return:  sets of features and the classifier
    """
    #phrasedoc
    ProcessedFeaures = General_preprocessing(arrayof_Features,False)


    # preprocessedtokens
    Removed_tokens_list = get_word_utility(ProcessedFeaures, 1)

    word_feature_list = get_word_utility(Removed_tokens_list, 2)

    print("processed", len(ProcessedFeaures))
    print("Removed_tokens_list", len(Removed_tokens_list))
    print("word_featurelist", len(word_feature_list))

    print("Train preprocessing")
    if(option ==1):
        print("NORMAL FEATURES-----------")
        sets_of_features = Create_featuresets(feature_list=word_feature_list, processed_features=ProcessedFeaures,tokens=Removed_tokens_list,option=1)
        #write_to_csv(sets_of_features,1)
        return sets_of_features , Classifier_process(sets_of_features, None,Classifier)

    elif(option ==2):
        print("SL FEATURES-----------")
        sets_of_features = Create_featuresets(feature_list=word_feature_list, processed_features=ProcessedFeaures,tokens=None,option= 2)
        #write_to_csv(sets_of_features, 2)
        return sets_of_features ,Classifier_process(sets_of_features,None,Classifier)
    elif(option ==3):
        print("NEGATION FEATURES-----------")
        sets_of_features = Create_featuresets(feature_list=word_feature_list, processed_features=ProcessedFeaures,tokens=None, option=3)
        #write_to_csv(sets_of_features, 3)
        return sets_of_features, Classifier_process(sets_of_features, None,Classifier)

    elif(option ==4):
        print("BIGRAM FEATURES--------------")
        sets_of_features = Create_featuresets(feature_list=word_feature_list,processed_features=ProcessedFeaures, tokens=Removed_tokens_list,option=4)
        #write_to_csv(sets_of_features, 4)
        return sets_of_features, Classifier_process(sets_of_features, None,Classifier)


def General_preprocessing(array_of_features, isTest):
    """
    raw preprocessing which includes tokenizing and converting to lower case
    :param array_of_features:
    :param isTest:
    :return: processed features
    """

    x = 0
    y = 0

    if isTest:
        x = 1
        y = 0
    else:
        x = 0
        y = 1



    ProcessedFeaures = []
    for entry in array_of_features:

        word = entry[x]
        sent = entry[y]
        tokenizer = RegexpTokenizer(r'\w+')
        entry[x] = Word_caseProcess(word)
        tokens = tokenizer.tokenize(word)
        if isTest:
            ProcessedFeaures.append([tokens,entry[y]])
        else:

            sentiment = int(entry[y])
            ProcessedFeaures.append([tokens, sentiment])
    return ProcessedFeaures



def test_preprocessing( arrayof_features, option):
    """
    quite similar to preprocessing method but it is tuned for test files as it does not have any sentiments
    :param arrayof_features:
    :param option:
    :return:
    """
    ProcessedFeaures = General_preprocessing(arrayof_features, True)


    # preprocessedtokens
    Removed_tokens_list = get_word_utility(ProcessedFeaures, 1)

    word_feature_list = get_word_utility(Removed_tokens_list, 2)
    print("processed", len(ProcessedFeaures))
    print("Removed_tokens_list", len(Removed_tokens_list))
    print("word_featurelist", len(word_feature_list))
    print("Test preprocessing")
    if (option == 1):
        print("NORMAL FEATURES-----------")
        sets_of_features = Create_featuresets(feature_list=word_feature_list, processed_features=ProcessedFeaures,tokens=Removed_tokens_list, option=1)
        return sets_of_features
    elif (option == 2):
        print("SL FEATURES-----------")
        sets_of_features = Create_featuresets(feature_list=word_feature_list, processed_features=ProcessedFeaures, tokens=None, option=2)
        return sets_of_features
    elif (option == 3):
        print("NEGATION FEATURES-----------")
        sets_of_features = Create_featuresets(feature_list=word_feature_list, processed_features=ProcessedFeaures,
                                              tokens=None, option=3)
        return sets_of_features

    elif (option == 4):
        print("BIGRAM FEATURES--------------")
        sets_of_features = Create_featuresets(feature_list=word_feature_list, processed_features=ProcessedFeaures,
                                              tokens=Removed_tokens_list, option=4)
        return sets_of_features





def Word_caseProcess(entry):
    """
    converting entries to lower case and recompile them
    :param entry:
    :return:
    """

    entry = entry.lower()
    sentence = entry.split('\s+')
    sentence = [re.compile(r'[-.?!/\%@,":;()|0-9]').sub("", w) for w in sentence]
    output_list = []
    for w in sentence:
        if w in ALL_THE_STOPWORDS:
            pass
        else:
            output_list.append(w)
    sentence = " ".join(output_list)
    return sentence


"""
Negation word process
1- Load_negaton_words
2- existance_of_word
3- Feature_list_Negation_process
4- Neg_preprocess
"""

def Load_negaton_words(Path = Negation_File):
    """
    load the negation file and process and return the array
    :param Path:
    :return:
    """
    try:
        f = open(Path,'r')

        temp = []
        for i in f:
            temp = i.split()

        return temp
    except:
        os.chdir(Path)
        get_lexiconFile(Path)

def existance_of_word(pointer, size,option):
    """

    checks mainly for Term presence
    :param pointer:
    :param size:
    :param option:
    :return:
    """

    if option ==1:
        if pointer+1 < size:
            return True
        else:
            return False
    elif option ==2:
        if pointer+3 < size:
            return True
        else:
            return False

def Feature_list_Negation_process(doc,feature_list, normalised_feature_list, negation_words ):
    """
    it checks with the negation words and see if the word in the doc has any likelihood to the negation words
    :param doc:
    :param feature_list:
    :param normalised_feature_list:
    :param negation_words:
    :return:
    """
    entry_size =len(doc)
    for entry in range(0, entry_size):
        myword = doc[entry]
        if (myword in negation_words) and existance_of_word(entry,entry_size,1):

            entry += 1
            existance = (doc[entry] in normalised_feature_list)
            feature_list['Holds(NOT{})'.format(doc[entry])] = existance
        else:
            if    existance_of_word(entry,entry_size,2) and ((myword.endswith('n')  and doc[entry+1] == "'" and doc[entry+2] == "t" )):
                entry += 3
                existance = (doc[entry] in normalised_feature_list)
                feature_list['Holds(NOT{})'.format(doc[entry])] = existance
            else:
                existance = (myword in normalised_feature_list)
                feature_list['contains({})'.format(myword)] = existance

    return feature_list

def Neg_preprocess( doc,normalised_feature_list, negation_words):
    """
    what needs to be called to process the negation process it calls thhe normal features for adding word presence directory
    :param doc:
    :param normalised_feature_list:
    :param negation_words:
    :return:
    """
    feature_list = normal_features(document= doc, word_features= normalised_feature_list,bigrams= None,option= 2)
    return Feature_list_Negation_process(doc, feature_list,normalised_feature_list, negation_words= negation_words)

"""
--------------------------------------------------
Bigram word process
BIGRAM_features
BIGRAM_process
"""
def BIGRAM_features(doc_tokens):
    """

    :param doc_tokens:
    :return:
    """
    features = nltk.BigramCollocationFinder.from_words(doc_tokens, window_size=3).nbest( nltk.collocations.BigramAssocMeasures().chi_sq, 3*1000)
    features = features[:5*100]
    return features

def BIGRAM_process(doc, word_feature, b_features):
    """

    :param doc:
    :param word_feature:
    :param b_features:
    :return:
    """

    normalised_features = normal_features(doc,word_feature,b_features, option= 3)
    return normalised_features



def submission_process(Classifier, test_data, option):
    """

    :param Classifier:
    :param test_data:
    :param option:
    :return:
    """



    print(len(test_data))
    n = 5
    tested_data = []
    for a in range(1, n+1):

        chunk = []
        if a == 1:
            print ("0 to ", int(len(test_data) / n))
            chunk = test_data[:int(len(test_data) / n)]
            tested_data.extend(test_preprocessing(chunk, option=option))



        else:
            print( (a - 1) * int(len(test_data) / n)," to ", a* int(len(test_data) / n))
            chunk = test_data[int((a - 1) * int(len(test_data) / n)): a * int(len(test_data) / n)]
            tested_data.extend(test_preprocessing(chunk, option=option))


    tested_data.extend(test_preprocessing(test_data[n * int(len(test_data) / n): len(test_data)],option))


    print("testing")

    Write_submission(tested_data, Classifier)




def Write_submission(test_data , Classifier ):
    """

    :param test_data:
    :param Classifier:
    :return:
    """

    myfile = open(SUBMISSION_FILE, "w")

    myfile.write("PhraseId" +','+"Sentiment"+'\n')
    print("writing to submissions")
    print("Test data size:", len(test_data))
    for (test,id) in test_data:

        myfile.write(str(id)+','+str(Classifier.classify(test))+'\n')
    myfile.close()


if __name__ == "__main__":
    arrayof_features, arrayof_Targets = convert_dataset("train.tsv", True, True)
    test_features = convert_dataset("test.tsv",False,True)

    #print(test_features)

    #arrayof_features = arrayof_features.tolist()
    #print (arrayof_features)
    #submission_process(train_data= arrayof_features, test_data= test_features)

    if (DEBUGGING_MODE):
        n = 4
        for j in range(1,n+1):
            Classifier = nltk.NaiveBayesClassifier
            for i in range(1,n+1):
                chunk = []
                if i == 1:
                    chunk  = arrayof_features[:int(len(arrayof_features)/n)]
                    feature_set, Classifier = preprocessing(chunk, j, Classifier)
                if i == n :
                    chunk = arrayof_features[int((i - 1) * int(len(arrayof_features) / n)): len(arrayof_features)]
                    feature_set, Classifier = preprocessing(chunk, j, Classifier)
                else:
                    chunk = arrayof_features[int((i-1)*int(len(arrayof_features)/n)): i * int(len(arrayof_features)/n)]
                    feature_set, Classifier = preprocessing(chunk, j, Classifier)
    else:
        n =4
        Classifier = nltk.NaiveBayesClassifier
        for a in range(1, n+1):
            chunk = []
            if a == 1:
                chunk = arrayof_features[:int(len(arrayof_features) / n)]
                feature_set, Classifier = preprocessing(chunk, 1, Classifier)
            if a == n:
                chunk = arrayof_features[
                        int((a - 1) * int(len(arrayof_features) / n)): a * len(arrayof_features)]
                feature_set, Classifier = preprocessing(chunk, 1, Classifier)
            else:
                chunk = arrayof_features[int((a - 1) * int(len(arrayof_features) / n)): a * int(len(arrayof_features) / n)]
                feature_set, Classifier = preprocessing(chunk, 1, Classifier)
        submission_process(Classifier, test_features, 1)