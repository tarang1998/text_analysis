#file: sentiment_mod.py


import nltk
import random
import pickle

#nltk.download("movie_review")

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC , NuSVC

from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize

from statistics import mode

#----------------------------------------------------------------------------------------
#??
class VoteClassifier(ClassifierI):

        #list of classifiers
	def __init__(self,*classifiers):
		self.classifiers=classifiers


        #features->  some text (includes if feature is present or not->  find_feature(text)) which needs to be classified
	def classify(self,features):
		votes=[]
		for c in self.classifiers:
			v=c.classify(features)
			votes.append(v)
		#print(votes)
		#print(mode(votes))
		return mode(votes)

	def confidence(self,features):
		votes=[]
		for c in self.classifiers:
			v=c.classify(features)
			votes.append(v)
		choice_votes=votes.count(mode(votes))#returns value of most frequent
		conf=(choice_votes/len(votes))*100
		return conf

#----------------------------------------------------------------------
	
#nltk.movie_review
#more time is taken to load the objects
#pickle to save the python objects(save the classifier so theres no need to train it again)

'''documents=[(list(movie_reviews.words(fileid)),category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fields(category)]
'''

#list of tuples
#tuple contains movie review (list of words) and category
document=[]

for category in movie_reviews.categories():
    #print(category)--> Yes No
    for fileid in movie_reviews.fileids(category):
        #print(fileid) neg/cv-636.txt
        document.append((list(movie_reviews.words(fileid)),category))

random.shuffle(document)
#print(document[1])

all_words=[]
for w in movie_reviews.words():
    #print(w)
    #remove stop words , Lemmatize
    all_words.append(w.lower())
   
#----------------------------------------------------------------------------------
'''
#external data

short_pos=open("short_review/positive.txt","r").read()
short_neg=open("short_review/negative.txt","r").read()

#document contains movie review string and category
document=[]

#all the words(both positive and negative)
all_words=[]

# J is adjective, R is adverb, V is verb,to remove unwanted stuff-->(alternative way to remove stopwords ,nos,dates)
allowed_word_types=["J","R","V"]

#allowed_word_type=["J"]

for r in short_pos.split("\n"):                             #sent_tokenize
        document.append((r,"pos"))
        words=word_tokenize(r)
        pos=nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())

for r in short_neg.split("\n"):
        document.append((r,"neg"))
        words=word_tokenize(r)
        pos=nltk.pos_tag(words)
        for w in pos:
                if w[1][0] in allowed_word_types:
                        all_words.append(w[0].lower())










        """
        all_words=[]
        short_pos_words=word_tokenize(short_pos)
        short_neg_words=word_tokenize(short_neg)

        for w in short_pos_words:
                all_words.append(w.lower())
        
        for w in short_neg_words:
                all_words.append(w.lower())
        """


'''
#----------------------------------------------------    

#count of each words
all_words=nltk.FreqDist(all_words)


#print(all_words.most_common(3000)) [ ',','a','the']
#print(all_words["stupid"])


#selct 3000 words from the list of all words   #[:5000]
word_features=list(all_words.keys())[:3000]
#word_features=list(all_words.keys())[:5000]

#save document
'''
saved_document=open("document_review.pickle","wb")
pickle.dump(document,saved_document)
saved_document.close()
'''

#save word_features
'''
saved_word_features=open("word_features.pickle","wb")
pickle.dump(word_features,saved_word_features)
saved_word_features.close()
'''


#load document
"""
document_f=open("document_review.pickle","rb")
document=pickle.load(document_f)
document_f.close()

# load feature words
feature_f=open("word_features.pickle","rb")
word_features=pickle.load(feature_f)
feature_f.close()
"""



#word_features=all_words.most_common(3000)
#print(word_features)

def find_features(document):
    #unique words
    words=set(document)#document ->list
    #words=word_tokenize(document) #document->string
    features={}
    for w in word_features:
        features[w]=(w in words) #return true or false value
    return features


#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featureset=[(find_features(rev),category) for (rev,category) in document]

#random.shuffle(featureset)
#training_set=featureset[:10000]
#testing_set=featureset[10000:]


training_set=featureset[:1900]
testing_set=featureset[1900:]

'''
#if not shuffled''
#positive testing data
training_set=featureset[:1900]
testing_set=featureset[1900:]

#negative testing data
training_set=featureset[100:]
testing_set=feautureset[:100]
'''

#---------------------------------------------------------------
#naive bayes classifier

classifier=nltk.NaiveBayesClassifier.train(training_set)


#to load the classifier
'''
classifier_f=open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_f)
classifier_f.close()
'''


print("Naive Bayes Algo accuracy:",(nltk.classify.accuracy(classifier,testing_set)*100))

classifier.show_most_informative_features(15)


#to save the classifier 
'''
save_classifier=open("naivebayes.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()
'''


#we could pickle each of the classifiers as there is no need to retrain them
#multinomialNB

MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)

'''
save_MNB_classifier=open("MNB.pickle","wb")
pickle.dump(MNB_classifier,save_MNB_classifier)
save_MNB_classifier.close()

MNB_classifier_f=open("MNB.pickle","rb")
MNB_classifier=pickle.load(MNB_classifier_f)
MNB_classifier_f.close()
'''


'''
#GaussianNB
GaussianNB_classifier=SklearnClassifier(GaussianNB())
GaussianNB_classifier.train(training_set)
print("GaussianNB_classifier accuracy percent:", (nltk.classify.accuracy(GaussianNB_classifier,testing_set))*100)
'''

#BernoulliNB
BernoulliNB_classifier=SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

'''
save_Bernoulli_classifier=open("Bernoulli.pickle","wb")
pickle.dump(BernoulliNB_classifier,save_Bernoulli_classifier)
save_Bernoulli_classifier.close()

Bernoulli_classifier_f=open("Bernoulli.pickle","rb")
BernoulliNB_classifier=pickle.load(Bernoulli_classifier_f)
Bernoulli_classifier_f.close()
'''


#LogisticRegression
LogisticRegression_classifier=SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

#SGDClassifier
SGDClassifier_classifier=SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

#SVC
SVC_classifier=SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)


#LinearSVC
LinearSVC_classifier=SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)


#NuSVC
NuSVC_classifier=SklearnClassifier(NuSVC( ))
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


voted_classifier=VoteClassifier(classifier,MNB_classifier,
                                BernoulliNB_classifier,
                                LogisticRegression_classifier,
                                SGDClassifier_classifier,SVC_classifier,
                                LinearSVC_classifier,NuSVC_classifier)




#print("voted_classifier accuracy percent:",(nltk.classify.accuracy(voted_classifier,testing_set))*100)
'''
print("Classification:",voted_classifier.classify(testing_set[0][0]),"Confidence %:",voted_classifier.confidence(testing_set[0][0]))
print("Classification:",voted_classifier.classify(testing_set[1][0]),"Confidence %:",voted_classifier.confidence(testing_set[1][0]))
print("Classification:",voted_classifier.classify(testing_set[2][0]),"Confidence %:",voted_classifier.confidence(testing_set[2][0]))
print("Classification:",voted_classifier.classify(testing_set[3][0]),"Confidence %:",voted_classifier.confidence(testing_set[3][0]))
print("Classification:",voted_classifier.classify(testing_set[4][0]),"Confidence %:",voted_classifier.confidence(testing_set[4][0]))

'''

def sentiment(text):
        features=find_features(text)
        return voted_classifier.classify(features), voted_classifier.confidence(features)












































