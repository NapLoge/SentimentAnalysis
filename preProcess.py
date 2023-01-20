import os

emojis = ["=]", '[=', ":)", ":-)", ";)", ";-)", ");", "=)", "8)", "):", ":o)", ";o)", "=o)", ":(", "(8", ":-(", "(=", "8(", "=(", ":}", ";d", ";p"]
breakLine = ["<br", "/><br", "/>"]  #Loose strings of line jump
forbiddenSymbols =['~', ':', '+', '[', '\\', '@', '^', '%', '(',  '"', '*', '|', ',', '&', '<', '`', '.', '_', '=', ']', '>', ';', '#', '$', ')', '/', '}', '{']
possibleSymbols = ["'", '-' , '!', '?']
vocabularyDictionary = {}
droppedWordsDictionary = {}   #Stores words in reviews that are not in vocabulary
numValidWordPos = 0
numValidWordNeg = 0


#Convert vocabulary into dicionary
def convertVocab(vocabFile):
  global vocabularyDictionary
  vocabfile = open(vocabFile, "r")
  vocab = vocabfile.readlines()
  for word in vocab:
    wordLen = len(word)
    if word[wordLen-1] == '\n': #Clean words that have jump of line
      word = word.replace("\n","")
    vocabularyDictionary[word] = 0
  vocabfile.close()


#Reads raw training and test file, call function to separate and eliminate garbage punctuation, makes one file for positive and another one for negative reviews
def readFile(folderName, outfile):
  dirName = os.listdir(folderName)
  outFile = open(outfile, "a")

  for fileName in dirName:
    filePath = os.path.join(folderName, fileName)
    file = open(filePath, "r")
    review = file.read().lower()
    review = review.split(" ")  #List of words in the review 
    reviewEdited = ""

    for word in review:
      wordEdited = ""
      if word in emojis:                      #Keeps emojis as is
        wordEdited = word + " "
      elif word in breakLine:                 #Removes loose strings of line jump
        wordEdited = " "
      else:                     #Any other word
        wordEdited = symbolProcessing(word)   #Removes some symbols or surround them with space
      reviewEdited += wordEdited + " "
    outFile.write(reviewEdited + "\n")      
  outFile.close()


#Removes lingering <br \> strings, punctuation at the beginning and/or end of word or separate them from words,
def symbolProcessing(word):
  #Removes breakline symbols that lingers at the end of words "hello.<br"
  if "<br" in word:
    lastIdx = len(word) - 1
    wordTemp = ""
    if (word[lastIdx-2] == "<" and word[lastIdx-1] == "b" and word[lastIdx] == "r"):
      wordTemp = word.replace("<br", " ")
      word = wordTemp

  #Removes breakline symbols that lingers at the beginning of word "/>Hello"
  if "/>" in word:
    wordTemp = ""
    if (word[0] == "/" and word[1] == ">"):
      wordTemp = word.replace("/>", " ")
      word = wordTemp

  #Removes symbols that are invalid at any position or separate possible symbols as ! and ? with blank space
  temp = ""
  for c in word:
    if c in forbiddenSymbols:
      temp += " "
    elif c == '!' or c == '?':
      temp += " " + c + " "
    else:
      temp += c
  wordList = temp.split()     #Eliminating symbols can add blank spaces into word string. Needs to ignore blank spaces

  word = ""
  for item in wordList:
    thisWord = item
    if "--" in item:
      thisWord = thisWord.replace("--", " ")

    if "''" in item:
      thisWord = thisWord.replace("''", " ")
    wordListTemp = thisWord.split()

    #Elminates valid symbols at the beginning of words as ' and -
    for itemTemp in wordListTemp:
      thisWordTemp = itemTemp
      if thisWordTemp[0] == "'" or thisWordTemp[0] == "-":
        thisWordTemp = thisWordTemp[1:len(thisWordTemp)]

      #Eliminates possessives 's and ', and - at end of words
      if len(thisWordTemp) >= 2:
        if thisWordTemp[len(thisWordTemp)-1] == "-":
          thisWordTemp = thisWordTemp[0:len(thisWordTemp)-1]

        if thisWordTemp[len(thisWordTemp)-2] == "'" and thisWordTemp[len(thisWordTemp)-1] == "s":
          thisWordTemp = thisWordTemp[0:len(thisWordTemp)-2]
        elif thisWordTemp[len(thisWordTemp)-1] == "'":
          thisWordTemp = thisWordTemp[0:len(thisWordTemp)-1]  

        thisWordTemp = thisWordTemp.replace(" ", "")

      word += thisWordTemp + " "
  return word


#Checks if words in training set and testing set are in vocabulary
def existWord(label, infile, outfile):
  global droppedWordsDictionary 
  global numValidWordPos
  global numValidWordNeg

  rawTraining = open(infile, "r")
  trainingData = rawTraining.readlines()
  trainingList = []

  for trainingReview in trainingData: #Each review
    trainingList = trainingReview.split(" ")
    reviewDict = {} #Keeps track of words in ONE review
    for word in trainingList:
      if not word.isnumeric():                  #Numbers don't belong in vocabulary
        if word in droppedWordsDictionary.keys():     #Have seen this invalid word
          droppedWordsDictionary[word] += 1
        else:                                   
          if word in vocabularyDictionary.keys():          #Word is in vocabulary
            numValidWordPos += 1
            vocabularyDictionary[word] += 1
            if word in reviewDict.keys():       #Word seen in this review previously
              reviewDict[word] += 1
            else:
              reviewDict[word] = 1
          else:
            droppedWordsDictionary[word] = 1          #Updates invalid words dictionary

    # method to construct vector and write in outfile
    constructVector(label, reviewDict, outfile)
  rawTraining.close()


def constructVector(label, reviewDict, outfile):
  outFile = open(outfile, "a")
  outVector = ""
  for word, counter in reviewDict.items():
    outVector += word + " " + str(counter) + " "
  if (label == "pos"):
    outVector = "+ " + outVector + "\n"
  else:
    outVector = "- " + outVector + "\n"
  
  outFile.write(outVector)
  outFile.close()

def cleaningVocabDict(wordToDelete):
  global vocabularyDictionary
  del vocabularyDictionary[wordToDelete]

##############################################################################################################
#Preprocess of training data
print("Running Preprocess \n")

#Store vocabulary in dictionary
convertVocab("imdb.vocab")
print("Size of vocabulary: " + str(len(vocabularyDictionary)) + "\n")

print("::::: Constructing Training Vector :::::\n")
outFilePOS = "trainingPosTemp.txt"
outfile = open(outFilePOS, "w")
readFile("train/pos", outFilePOS)
outfile.close

outFileNEG = "trainingNegTemp.txt"
outfile = open(outFileNEG, "w")
readFile("train/neg", outFileNEG)
outfile.close

#Creates processed training data for NB algorithn
outFileTRAINING = "trainingVectors.txt"
outfile = open(outFileTRAINING, "w")
existWord("pos", outFilePOS, outFileTRAINING)
existWord("neg", outFileNEG, outFileTRAINING)
outfile.close()
os.remove(outFilePOS)
os.remove(outFileNEG)

print("::::: Constructing Testing Vector :::::\n")
outFilePOS = "testingPosTemp.txt"
outfile = open(outFilePOS, "w")
readFile("test/pos", outFilePOS)
outfile.close()

outFileNEG = "testingNegTemp.txt"
outfile = open(outFileNEG, "w")
readFile("test/neg", outFileNEG)
outfile.close()

outFileTESTING = "testingVectors.txt"
outfile = open(outFileTESTING, "w")
existWord("pos", outFilePOS, outFileTESTING)
existWord("neg", outFileNEG, outFileTESTING)
outfile.close()
os.remove(outFilePOS)
os.remove(outFileNEG)


##############################################################################################################
print("\n\n::::: Exploring STOP words removal features :::::\n\n")
sortedVocab = list(sorted(vocabularyDictionary.items(), key=lambda item: item[1]))
vocabSize = len(vocabularyDictionary)
out = open("stopWord.txt", "w")

for idx in range(vocabSize-100, vocabSize):
  out.write(sortedVocab[idx][0] + "\n")
  cleaningVocabDict(sortedVocab[idx][0])
out.close()

##############################################################################################################
#Preprocess of training data
print("Running Preprocess \n")
print("Size of vocabulary: " + str(len(vocabularyDictionary)) + "\n")

# vocabularyDictionary = {}
droppedWordsDictionary = {}   #Stores words in reviews that are not in vocabulary
numValidWordPos = 0
numValidWordNeg = 0

print("::::: Constructing Training Vector with Stop Word :::::\n")
outFilePOS = "modifiedTrainingPosTemp.txt"
outfile = open(outFilePOS, "w")
readFile("train/pos", outFilePOS)
outfile.close()

outFileNEG = "modifiedTrainingNegTemp.txt"
outfile = open(outFileNEG, "w")
readFile("train/neg", outFileNEG)
outfile.close()

#Creates processed training data for NB algorithn
outFileTRAINING = "modifiedTrainingVectors.txt"
outfile = open(outFileTRAINING, "w")
existWord("pos", outFilePOS, outFileTRAINING)
existWord("neg", outFileNEG, outFileTRAINING)
outfile.close()
os.remove(outFilePOS)
os.remove(outFileNEG)

print("::::: Constructing Testing Vector with Stop Word :::::\n")
outFilePOS = "modifiedTestingPosTemp.txt"
outfile = open(outFilePOS, "w")
readFile("test/pos", outFilePOS)
outfile.close()

outFileNEG = "modifiedTestingNegTemp.txt"
outfile = open(outFileNEG, "w")
readFile("test/neg", outFileNEG)
outfile.close()

outFileTESTING = "modifiedTestingVectors.txt"
outfile = open(outFileTESTING, "w")
existWord("pos", outFilePOS, outFileTESTING)
existWord("neg", outFileNEG, outFileTESTING)
outfile.close()
os.remove(outFilePOS)
os.remove(outFileNEG)