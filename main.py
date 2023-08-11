import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from preprocessing import pre_processing

df = pd.read_csv("file.tsv", sep='\t')
df.columns = ['label', 'tweets']
label = df['label']
data = df['tweets']
df.insert(1, "punct%", df['tweets'].apply(lambda x: pre_processing.count_punct(x)))
df.insert(1, "Tweet_len", df['tweets'].apply(lambda x: len(x) - x.count(' ')))
print(df.head())
tfidf = TfidfVectorizer(ngram_range=(1, 2))
features = tfidf.fit_transform(data)
print(features.shape)
print('Sparse Matrix:\n', features)
features = pd.DataFrame(features.toarray())
features.columns = tfidf.get_feature_names_out()
print(features)

result = pre_processing.clean_data(data)
pre_processing.display(result)

vect = TfidfVectorizer()
result = vect.fit_transform(result)
##############################################
label_encoder = preprocessing.LabelEncoder()
label = label_encoder.fit_transform(label)
# display(result)

kf = KFold(n_splits=20)
for train_index, test_index in kf.split(result):
    X_train, X_test, y_train, y_test = result[train_index], result[test_index], label[train_index], label[test_index]
# using logistic
scores_logistic = LogisticRegression(solver='liblinear', C=10, random_state=1)
scores_logistic.fit(X_train, y_train)
p = scores_logistic.predict(X_test)
score = accuracy_score(y_test, p)
print("Using Logistic : ")
print(score)

# using svm
sv = svm.SVC(kernel='linear')  # Linear Kernel
sv.fit(X_train, y_train)
y_pred = sv.predict(X_test)
print("Using SVM : ")
print(metrics.accuracy_score(y_test, y_pred))

# using decision tree
dt = DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("Using DT:")
print(metrics.accuracy_score(y_test, y_pred))
