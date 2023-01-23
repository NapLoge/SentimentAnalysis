
# Sentiment Analysis 

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
that do not belong in the valid vocabulary file (provided), and converting them into word vector format *(Class, Word0, Count0, ... , WordN, CountN)*. 

***Positive example:*** “She was at the store while I was at the movies”

***Vector:*** +, she, 1, was, 2, at, 2, the, 2, store, 1, while, 1, i, 1, movies, 1

The prediction accuracy achieved on the testing data set is 81.32% (takes only into account words 
that are known in the training data). Additionally, the elimination of stop words was also included 
in the model to compare their accuracies, a rate of 82.484% was achieved by adding this feature.

The stop words are the top 100 most frequently seen words in the training data set. Their occurrances
are not taken into account for calculations, and a list of them can be found in "stopWord.txt"

## Steps:

1. Download the repository

2. Run pre-processing script
    ```console
    naploge@machine:~$ python3 preProcess.py
    ```
    Note: Once preProcess.py finishes, four files are created.
    * trainingVectors.txt file: stores training reviews in vector format after invalid words removal.

    * testingVectors.txt file: stores testing reviews in vector format after invalid words removal.

    * modifiedTrainingVector.txt file: stores training reviews in vector format after invalid words and stop words removal.

    * modifiedTestingVector.txt file: stores testing reviews in vector format after invalid words and stop words removal.

3. Run sentiment analysis script
    ```console
    naploge@machine:~$ python3 naiveBayesBOW.py trainingVectors.txt testingVectors.txt parameter.txt output.txt
    ```
    Note: Two result files are created.
    * parameter.txt file: stores the parameters of the model.

    * output.txt file: stores the outputs of the model

4. (Optional) Run sentiment analysis script with stop words removal
    ```console
    naploge@machine:~$ python3 naiveBayesBOW.py modifiedTrainingVectors.txt modifiedTestingVectors.txt modifiedParameter.txt modifiedOutput.txt 
    ```
    Note: This last step runs the sentiment analysis with stop word removal feature. It uses the modified training and testing vectors. 
