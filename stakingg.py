import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier
from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import gensim
from gensim.models import Word2Vec

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

from newspaper import Article
from sklearn.model_selection import RandomizedSearchCV

import pickle 
# Đọc dữ liệu vào dataframe
df = pd.read_csv(r"C:\Users\dell\Downloads\train.csv (1)\news.csv")
# Liệt kê các giá trị thiếu
missing_values_total = df.isnull().sum().sum()
missing_values = df.isnull().sum()
# Tính tổng số giá trị thiếu trong mỗi cột
missing_values = df.isnull().sum()

# Lọc chỉ các cột có giá trị thiếu
missing_values = missing_values[missing_values > 0]

# Tổng số giá trị thiếu trên toàn bộ DataFrame
missing_values_total = missing_values.sum()

# Hiển thị thông tin giá trị thiếu
print("Tổng giá trị thiếu:", missing_values_total)
print("Các cột có giá trị thiếu:")
print(missing_values)

# Thay thế các giá trị null bằng khoảng trống
df = df.fillna('')

# Kiểm tra lại số lượng giá trị thiếu sau khi thay thế
print("Số lượng giá trị thiếu sau khi thay thế:")
print(df.isnull().sum())

# Thay đổi các nhãn của dataframe
df["label"] = df["label"].astype("object")
df.loc[(df["label"] == 1), ["label"]] = "FAKE"
df.loc[(df["label"] == 0), ["label"]] = "REAL"

# Lấy các nhãn từ dataframe
labels = df.label

# Tạo cột mới hợp nhất giá trị của cột 'title','author' và 'text'
df['combined_info'] = df['author'] + ' ' + df['title'] + ' ' + df['text']
# Xem cột mới
df["combined_info"].head()

# Chuyển thành chữ thường, loại bỏ các dấu câu
df['combined_info'] = df['combined_info'].str.lower().replace(f'[{string.punctuation}]', '', regex=True) 
df['combined_info'].head()

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(df["combined_info"], df['label'], test_size=0.2, random_state=42)

# Tạo TfidfVectorizer và chuyển đổi dữ liệu
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# RandomizedSearchCV cho PassiveAggressiveClassifier
pac_params = {'C': [0.01, 0.1, 1, 10], 'max_iter': [100, 500], 'tol': [1e-3, 1e-4]}
pac = PassiveAggressiveClassifier()
pac_search = RandomizedSearchCV(pac, pac_params, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
pac_search.fit(X_train_tfidf, y_train)
best_pac = pac_search.best_estimator_

# RandomizedSearchCV cho Logistic Regression
lr_params = {'C': [0.1, 1, 10], 'solver': ["liblinear", "lbfgs"]}
lr = LogisticRegression(max_iter=1000)
lr_search = RandomizedSearchCV(lr, lr_params, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
lr_search.fit(X_train_tfidf, y_train)
best_lr = lr_search.best_estimator_

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[('pac', best_pac), ('lr', best_lr)],
    final_estimator=LogisticRegression()
)
stacking_clf.fit(X_train_tfidf, y_train)

# Đánh giá hiệu suất
y_pred_stacking = stacking_clf.predict(X_test_tfidf)
print("Test set evaluation for Stacking Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_stacking))
print(classification_report(y_test, y_pred_stacking, target_names=["FAKE", "REAL"]))

# Lưu mô hình và vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('stacking_model.pkl', 'wb') as f:
    pickle.dump(stacking_clf, f)
