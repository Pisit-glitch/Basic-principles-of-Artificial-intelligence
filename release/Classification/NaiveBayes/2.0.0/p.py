import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data_path = 'updated_cleaned_patient_data.csv'
data = pd.read_csv(data_path)

feature_cols = [
    'FBS', 'BMI', 'Diabetes', 'age', 'hypertension',
    'vegetarian (1= yes, 0=no)', 'Exercise (min/week)',
    'Cholesterol', 'Living_Area_Code', 'Marriage_Status_Code'
]

data['text'] = data[feature_cols].astype(str).agg(' '.join, axis=1)

X = data['text']
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1,2),
    min_df=3
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB(alpha=0.1)

model.fit(X_train_tfidf, y_train)

train_pred = model.predict(X_train_tfidf)
test_pred = model.predict(X_test_tfidf)

print("Train Accuracy :", accuracy_score(y_train, train_pred))
print("Test Accuracy :", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))