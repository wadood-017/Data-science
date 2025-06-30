import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# loading the dataset
df=pd.read_csv("train.csv")

# separating catagorical and numerical colunmns
cat_columns=df.select_dtypes(include="object").columns
num_columns=df.select_dtypes(exclude='object').columns


# fill missing values
for columns in cat_columns:
    df[columns].fillna(df[columns].mode()[0], inplace=True)

for col in num_columns:
    df[col].fillna(df[col].median(), inplace=True)


# visualizations
sns.boxplot(data=df, x='Education', y='LoanAmount')
plt.title("Loan Amount by Education")
plt.show()

sns.boxplot(data=df, x='Gender', y='ApplicantIncome')
plt.title("Applicant Income by Gender")
plt.show()


# encoding in labels
le = LabelEncoder()
for col in cat_columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# test and train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# predictions
y_pred = log_model.predict(X_test)

# evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))