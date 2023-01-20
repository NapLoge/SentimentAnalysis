import math
import sys

probPositive = 0           #Tracks probability of positive/negative review
probNegative = 0 
posReviewCount = 0         #Tracks number of positive/negative reviews in training
negReviewCount = 0
posCorpusCount = 0         #Tracks total number of words in positive/negative reviews in training
negCorpusCount = 0
posWordDictionary = {}     #Dictionary of valid words in positive/negative reviews in training
negWordDictionary = {}
wordVocabulary = set()     #Tracks vocabulary of training set positive and negative
posNB = {}                 #Naive Bayes parameters (word's log10 likelihood with smoothing) for positive/negative reviews - expressed in log10
negNB = {}                 
reviewPredictions = []     #Tracks review type, positive probability, negative probability, model prediction

correctPredictionCount = 0
wrongPredictionCount = 0
samePredictionCount = 0

posReviewsIdentified = 0
negReviewsIdentified = 0
posReviewsIdentifiedWrong = 0
negReviewsIdentifiedWrong = 0

def readFile(dataType, infile):
  global probPositive
  global probNegative 
  reviewSet = infile.readlines()

  if dataType == "training":  
    for workingLine in reviewSet:
      workingLine = workingLine.replace("\n", "")
      wordList = workingLine.split()
      reviewCounter(wordList[0])                    #Tracks number of reviews read
      for index in range(1,len(wordList),2):        #(type, word, count)
        buildDictionary(wordList[0], wordList[index], wordList[index+1])
    probPositive = math.log10( posReviewCount / (posReviewCount + negReviewCount))
    probNegative = math.log10( negReviewCount / (posReviewCount + negReviewCount))

  elif dataType == "testing":
    for workingLine in reviewSet:
      workingLine = workingLine.replace("\n", "")
      wordList = workingLine.split()
      probabilityCalculation(wordList)

#Method creates a dictionary for words in positive reviews and another for words in negative reviews with the counter of each word
def buildDictionary(reviewType, word, count):
  global posWordDictionary
  global negWordDictionary
  global wordVocabulary
  global posCorpusCount
  global negCorpusCount

  #Build a dictionary of positive reviews / dictionary of negative reviews and their word count (corpus)
  if reviewType == "+":
    posCorpusCount += int(count)
    if word in posWordDictionary.keys():
      posWordDictionary[word] += int(count)
    elif not word in posWordDictionary.keys():
      posWordDictionary[word] = int(count)
  elif reviewType == "-":
    negCorpusCount += int(count)
    if word in negWordDictionary.keys():
      negWordDictionary[word] += int(count)
    elif not word in negWordDictionary.keys():
      negWordDictionary[word] = int(count)

  #Build set of vocabulary in both positive and negative reviews - vocabulary of model
  wordVocabulary.add(word)   


def parameterCalculation():
  global posNB
  global negNB

  for word in wordVocabulary:
    posNB[word] = smoothedNB("pos", word, posCorpusCount)
    negNB[word] = smoothedNB("neg", word, negCorpusCount)


#Loglikelihood with smoothing of word for positive and negative class
def smoothedNB(reviewType, word, corpusCount):
  vocabSize = len(wordVocabulary)
  count = 0

  if reviewType == "pos" and (word in posWordDictionary):
    count = posWordDictionary[word]
  elif reviewType == "neg" and (word in negWordDictionary):
    count = negWordDictionary[word]

  return math.log10((count + 1) / (corpusCount + vocabSize))


#Calculates the number of positive and negative reviews
def reviewCounter(reviewType):
  global posReviewCount
  global negReviewCount

  if reviewType == "+":
    posReviewCount += 1
  elif reviewType == "-":
    negReviewCount += 1


def probabilityCalculation(wordList):
  global correctPredictionCount
  global wrongPredictionCount
  global samePredictionCount
  global posReviewsIdentified
  global negReviewsIdentified 
  global posReviewsIdentifiedWrong
  global negReviewsIdentifiedWrong
  global reviewPredictions

  posReview = 1
  negReview = 1

  for index in range(1,len(wordList),2):    #list of word of each review
    if wordList[index] in wordVocabulary:   #Include prob only if word is present in training
      posReview += (float(posNB[wordList[index]]) * int(wordList[index+1]))
      negReview += (float(negNB[wordList[index]]) * int(wordList[index+1]))
    
  posReview += probPositive
  negReview += probNegative
  prediction = modelPrediction(posReview, negReview)
  
  if prediction == wordList[0]:
    correctPredictionCount += 1
    if wordList[0] == "+":
      posReviewsIdentified += 1
    else:
      negReviewsIdentified += 1
  elif prediction == "=":
    samePredictionCount += 1
  else:
    wrongPredictionCount += 1
    if wordList[0] == "+":
      posReviewsIdentifiedWrong += 1
    else:
      negReviewsIdentifiedWrong += 1
  reviewPredictions.append((wordList[0], posReview, negReview, prediction))


def modelPrediction(posClassProb, negClassProb):
  if posClassProb > negClassProb:
    return "+"
  elif posClassProb < negClassProb:
    return "-"
  else:     #Equal probability of being positive and negative
    return "="


#Calculates Naive Bayes Model sentiment analysis prediction accuracy
def accuracyCalculation():
  modelAccuracy = correctPredictionCount * 100 / (correctPredictionCount + wrongPredictionCount + samePredictionCount)
  return modelAccuracy


def printModelParameters(outfile):
  outfile.write("Positive reviews count: " + str(posReviewCount) + "\n\n")
  outfile.write("Negative reviews count: " + str(negReviewCount) + "\n\n")
  outfile.write("Number words in positive reviews: " + str(posCorpusCount) + "\n\n")
  outfile.write("Number words in negative reviews: " + str(negCorpusCount) + "\n\n")
  outfile.write("\n\nThe size of the training Vocabulary: " + str(len(posNB)) + " words \n\n")
  outfile.write("The Logarithmic in base 10 of probability of positive review: " + str(probPositive) + "\n\n")
  outfile.write("The Logarithmic in base 10 of probability of negative review: " + str(probNegative) + "\n\n\n")
  outfile.write("Logarithmic in base 10 of the likelihood of each word in vocabulary \n")
  outfile.write("WORD                                                         POSITIVE LIKELIHOOD        NEGATIVE LIKELIHOOD \n\n")
  for word, param in posNB.items():
    spacing = spacingFormat(60, word)  
    outfile.write(word + spacing + str(param) + "         " + str(negNB[word]) + "\n")


#Spacing formatting
def spacingFormat(offset, word):
  spaceSize = offset - len(word)
  space = ""
  for i in range(spaceSize):
    space += " "
  return space


def printModelOutput(outfile):
  outfile.write("Total number of reviews:  " + str(correctPredictionCount + wrongPredictionCount) + "\n\n")
  outfile.write("Number of correct predictions:  " + str(correctPredictionCount) + "\n\n")
  outfile.write("Number of wrong predictions: " + str(wrongPredictionCount) + "\n\n")
  outfile.write("Number of undetermined predictions (equal probability): " + str(samePredictionCount) + "\n\n")
  
  if posReviewsIdentified + posReviewsIdentifiedWrong > 0:
    outfile.write("Number of positive reviews correctly identified: " + str(posReviewsIdentified) + "\n\n")
    outfile.write("Number of positive reviews wrongly identified: " + str(posReviewsIdentifiedWrong) + "\n\n")
    outfile.write("NB BOW Accuracy for positive reviews: " + str(posReviewsIdentified * 100 / (posReviewsIdentified + posReviewsIdentifiedWrong)) + "%\n\n")

  if negReviewsIdentified + negReviewsIdentifiedWrong > 0:
    outfile.write("Number of negative reviews correctly identified: " + str(negReviewsIdentified) + "\n\n")
    outfile.write("Number of negative reviews wrongly identified: " + str(negReviewsIdentifiedWrong) + "\n\n")
    outfile.write("NB BOW Accuracy for negative reviews: " + str(negReviewsIdentified * 100 / (negReviewsIdentified + negReviewsIdentifiedWrong)) + "%\n\n")

  outfile.write("Naive Bayes Bag of Words Accuracy: " + str(accuracy) + "% \n \n")

  outfile.write("List of review type, log10 of probabilitues and model prediction \n")
  outfile.write("Review type       Prediction        Log10(Positive Prob)        Log10(Negative Prob)\n")
  for idx in range(len(reviewPredictions)):
    outfile.write(reviewPredictions[idx][0] + "                 " + reviewPredictions[idx][3] + "                " + str(reviewPredictions[idx][1]) + "          " + str(reviewPredictions[idx][2]) + "\n")


########################################################################################
print("::::: Start Training NB Model with training data ::::: \n")

trainingTxt = open(sys.argv[1], "r")    #Contains review vectors from pre processing
testingTxt = open(sys.argv[2], "r")
parameterTxt = open(sys.argv[3], "w")        #Store model parameters
outputTxt = open(sys.argv[4], "w")

readFile("training", trainingTxt)

parameterCalculation()                            #Param. store in posNB and negNB

print("::::: Start Testing Process ::::: \n")

readFile("testing", testingTxt)

accuracy = accuracyCalculation()

printModelParameters(parameterTxt)

printModelOutput(outputTxt)

trainingTxt.close()
testingTxt.close()
parameterTxt.close()
outputTxt.close()