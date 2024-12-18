import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

data = pd.read_csv('data/train.csv')
data = data.drop(columns=['Cabin'])
data['Age'] = SimpleImputer(strategy='mean').fit_transform(data[['Age']])
data = data.drop_duplicates(subset=['PassengerId'])
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

x = data.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Fare', 'Embarked'])
y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

models = [
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=3),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    XGBClassifier(),
    AdaBoostClassifier()
]

best_model, best_f1 = None, 0

for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1, best_model = f1, model

print(f"Best Model: {best_model.__class__.__name__}, F1 Score: {best_f1:.2f}")

import pickle

with open('titanic_survivor_prediction.pkl', 'wb') as file:
    pickle.dump(best_model, file)
