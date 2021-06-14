from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import pandas as pd
from imblearn.combine import SMOTETomek
import seaborn as sns
import matplotlib.pyplot as plt




def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
  '''
  Compute the ROCAUC of a multi-class test set. 
  
  Return: (arr) ROCAUC of each class.
  '''

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict


# Load data
dataset = pd.read_csv("glass.csv")
print(dataset.head())

# Check missing data
print(dataset.isnull().sum())

# Feature selection by observing the covariance.
corr = dataset.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()


# Check size
print(dataset.shape)

# Split the data into input and output
X = dataset.iloc[:,0:9]
Y = dataset['Type']
print(X)
print(Y)

# Check if the data are balanced
for y in set(Y):
    print('{}....{}'.format(y,len(Y[Y==y])))

# Oversample the imbalanced data using SMOTETomek
smote = SMOTETomek(random_state = 42)
X_over, Y_over = smote.fit_resample(X, Y)

# Observe the oversampled data
for y in set(Y_over):
    print('{}....{}'.format(y,len(Y_over[Y_over==y])))

# Split the oversampling data into training and test data.
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X_over, Y_over, test_size=test_size, random_state=seed)

print(X_train.shape)
print(X_test.shape)



xgb = XGBClassifier(learning_rate = 0.1, n_estimators = 300, max_depth = 5)
xgb.fit(X_train, y_train)
y_xgb = xgb.predict(X_test)
print('XGBoost Accuracy' , accuracy_score(y_test, y_xgb)  )
print('XGBoost ROCAUC' , roc_auc_score_multiclass(y_test, y_xgb)  )

