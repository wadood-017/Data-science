import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# loading the dataset
df=pd.read_csv("Churn_Modelling.csv")

# dropping irrelevent columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# encoding values
# label encode gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])


df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# preparing targets
X = df.drop('Exited', axis=1)
y = df['Exited']


# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)


# evalutation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()