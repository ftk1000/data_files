https://www.kaggle.com/c/spooky-author-identification

# Spooky Author Identification
Share code and discuss insights to identify horror authors from their writings


## Data
https://www.kaggle.com/c/spooky-author-identification/data


Submissions are evaluated using multi-class logarithmic loss. Each id has one true class. For each id, you must submit a predicted probability for each author. The formula is then:

logloss=−1N∑i=1N∑j=1Myijlog(pij),
where N is the number of observations in the test set, M is the number of class labels (3 classes), log is the natural logarithm, yij is 1 if observation i belongs to class j and 0 otherwise, and pij is the predicted probability that observation i belongs to class j.

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, predicted probabilities are replaced with max(min(p,1−10−15),10−15).

Submission File
You must submit a csv file with the id, and a probability for each of the three classes.
The order of the rows does not matter. The file must have a header and should look like the following:


      id,EAP,HPL,MWS
      id07943,0.33,0.33,0.33
