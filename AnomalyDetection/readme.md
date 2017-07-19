# Anomaly Detection Examples

We will explore a several anomaly detection techniques, as described by the survey paper by Chandola et. al. 

---

## Description of Dataset ##

We will use the [Credit Card Fraud Detection](https://www.kaggle.com/dalpozz/creditcardfraud). As described by Kaggle, 

> "the datasets contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise."

As said before, we will consider semi-supervised techiques, were we assume that the training data has labeled instances for only the normal class. In this case, we typically build a model to capture only the normal behavior, tune the model complexity in a labeled validation set and evaluate the model in a equally labeled testing set. More formally, during training, we only model ( P(x) ), that is, the probability of the normal data. At test time, we classify as an anomaly instances that have very low probability. Even when such probability is not directly available in the model, we can, most of the time define a score function proportional to it. Using a validation set, we can tune a threshold for the probability or score below which we will classify the instance as an anomaly.

For more detailed explanation about the semi-supervised approach, please refer to the slides.

Since some of the techniques considered works (much) better if the features are all Gaussian, we will transform the Amount and Time features to make them more Gaussian like. Our approach will be to simply take the ( log ) transformation of them. Since some features have zeros, we first add one to ( log ) of zero.

---

```python
dataset = pd.read_csv('../input/creditcard.csv')#.drop('Time', axis=1)

dataset['Amount'] = np.log(dataset['Amount'] + 1)
dataset['Time'] = np.log(dataset['Time'] + 1)
normal = dataset[dataset['Class'] == 0]

anomaly = dataset[dataset['Class'] == 1]
print(normal.shape)
print(anomaly.shape)

```

To split the data, we will hold 50% of the normal instances for training. The rest of the normal data will be equally split among a validation and a test set. The abnormal instances will also be equally spitted between the validation and test set.

```python
from sklearn.model_selection import train_test_split

train, normal_test, _, _ = train_test_split(normal, normal, test_size=.2, random_state=42)

normal_valid, normal_test, _, _ = train_test_split(normal_test, normal_test, test_size=.5, random_state=42)
anormal_valid, anormal_test, _, _ = train_test_split(anomaly, anomaly, test_size=.5, random_state=42)

train = train.reset_index(drop=True)
valid = normal_valid.append(anormal_valid).sample(frac=1).reset_index(drop=True)
test = normal_test.append(anormal_test).sample(frac=1).reset_index(drop=True)

print('Train shape: ', train.shape)
print('Proportion os anomaly in training set: %.2f\n' % train['Class'].mean())
print('Valid shape: ', valid.shape)
print('Proportion os anomaly in validation set: %.2f\n' % valid['Class'].mean())
print('Test shape:, ', test.shape)
print('Proportion os anomaly in test set: %.2f\n' %
```

Since the data is heavily unbalanced, there is no point in using simple accuracy as evaluation metrics, since a naive model that always predicts a normal class would get more than 99% accuracy. We will thus use precision and recall metrics (Wiki F1 score)<https://en.wikipedia.org/wiki/F1_score>


Precision ( ( P) ) is defined as the number of true positives ( ( T_p) ) over the number of true positives plus the number of false positives (( F_p)):
P=TpTp+Fp
P=TpTp+Fp
 
Recall (( R )) is defined as the number of true positives (( T_p )) over the number of true positives plus the number of false negatives (( F_n )):
R=TpTp+Fn
R=TpTp+Fn
 
These quantities are also related to the (( F_1 )) score.





Precision ( ( P) ) is defined as the number of true positives ( ( T_p) ) over the number of true positives plus the number of false positives (( F_p)):
P=TpTp+Fp
P=TpTp+Fp
 
Recall (( R )) is defined as the number of true positives (( T_p )) over the number of true positives plus the number of false negatives (( F_n )):
R=TpTp+Fn
R=TpTp+Fn
 
These quantities are also related to the (( F_1 )) score."






Precision ( ( P) ) is defined as the number of true positives ( ( T_p) ) over the number of true positives plus the number of false positives (( F_p)):
P=TpTp+Fp
P=TpTp+Fp
 
Recall (( R )) is defined as the number of true positives (( T_p )) over the number of true positives plus the number of false negatives (( F_n )):
R=TpTp+Fn
R=TpTp+Fn
 
These quantities are also related to the (( F_1 )) score."
