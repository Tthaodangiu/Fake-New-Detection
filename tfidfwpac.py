import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import string

import pickle  # Import pickle để lưu mô hình

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
# Chia bộ dữ liệu thành train_set và test_set
X_train, X_test, y_train, y_test = train_test_split(df["combined_info"], df['label'], test_size=0.2, random_state=42)

# Khởi tạo TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

# Huấn luyện và biến đổi tập huấn luyện, biến đổi tập kiểm tra
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Xem 2 dòng đầu tiên của X_train_tfidf
print("X_train_tfidf:", X_train_tfidf[:2])
# Xem 2 dòng đầu tiên của X_test_tfidf
print("X_test_tfidf:", X_test_tfidf[:2])

# Khởi tạo PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=100)
# Huấn luyện mô hình trên tập dữ liệu đã chuyển đổi bằng TF-IDF
pac.fit(X_train_tfidf, y_train)
# Dự đoán trên tập test và tính toán độ chính xác
y_pred_tfidf1 = pac.predict(X_test_tfidf)
score = accuracy_score(y_test, y_pred_tfidf1)
print(f"Accuracy: {round(score*100,2)}%")

# Lưu lại vectorizer và mô hình nếu cần
with open('fake_news_pac_model.pkl', 'wb') as model_file:
    pickle.dump(pac, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

