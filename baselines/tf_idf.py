from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report

from sklearn.model_selection import train_test_split

def model(df, y, y_label, type):
    
    X = df[type]
    y = y  # z MultiLabelBinarizer

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        stop_words='english'
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("F1 micro:", f1_score(y_test, y_pred, average='micro'))
    print("F1 macro:", f1_score(y_test, y_pred, average='macro'))

    print(classification_report(y_test, y_pred, target_names=y_label))