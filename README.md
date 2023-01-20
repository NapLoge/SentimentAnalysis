Sentiment Analysis 

The task of text categorization or sentiment analysis relates to extracting 
the positive or negative orientation that the writer expresses regarding 
some topic. In this case, we classify texts related to movie reviews as positive 
or negative impressions.

The classification method implemented is the Naive Bayes approach, which relies in
representing a document as a bag of words or an unordered set of words where their 
number of occurrences but not their positions are taken into consideration. 

The training of the model is performed using a data set of 25000 movie reviews composed
by (50-50) positive and negative reviews. The testing data set has similar characteristics.
These movie reviews were provided as raw data that needed pre-processing to separate line breaks,
punctuation symbols, emojis, possessive 's and s', etc. As well as eliminating words 
that do not belong in the valid vocabulary file (provided). 

The prediction accuracy achieved on the testing data set is 81.32% (takes only into account words 
that are known in the training data). Additionally, the elimination of stop words was also included 
in the model to compare their accuracies, a rate of 82.484% was achieved by adding this feature.

The stop words are the top 100 most frequently seen words in the training data set. Their occurrances
are not taken into account for calculations, and a list of them can be found in "stopWord.txt"

Steps
1 ) Download the entire repository

2 ) Run "python3 preProcess.py"
    Note: Once preProcess.py finishes, four files have been created.
          "trainingVectors.txt" and "testingVectors.txt" result from preprocessing the raw data against vocabulary
          "modifiedTrainingVector.txt" and "modifiedTestingVector.txt" result from removing the stop words

3 ) Run "python3 naiveBayesBOW.py trainingVectors.txt testingVectors.txt parameter.txt output.txt"
    Note: parameter.txt is the name of document to be created that stores the parameters of the model
          output.txt is the name of document to be created that stores the outputs of the model

4 ) ADDITIONAL FEATURE -> Run "python3 naiveBayesBOW.py modifiedTrainingVectors.txt modifiedTestingVectors.txt modifiedParameter.txt modifiedOutput.txt"
    Note: This last step runs the sentiment analysis with stop word removal feature. 
          modifiedParameter.txt contains model's parameters after stop words removal
          modifiedOutput.txt contains model's output after stop words removal
