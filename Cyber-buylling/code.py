import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import re
from nltk.tokenize import sent_tokenize
from nltk import download
from textblob import TextBlob

# 确保下载punkt数据
download('punkt')

# 数据加载和预处理
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

data = pd.read_json('Dataset for Detection of Cyber-Trolls.json', lines=True)
data['cleaned_content'] = data['content'].apply(preprocess)
data['label'] = data['annotation'].apply(lambda x: x['label'][0])

# 添加文本长度作为一个特征
data['length'] = data['cleaned_content'].apply(len)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data[['cleaned_content', 'length']], data['label'], test_size=0.2, random_state=42)

# TF-IDF向量化 + 文本长度合并
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['cleaned_content'])
X_train_tfidf = np.hstack((X_train_tfidf.toarray(), X_train[['length']].values / X_train['length'].max()))  # Normalize length and combine

X_test_tfidf = tfidf_vectorizer.transform(X_test['cleaned_content'])
X_test_tfidf = np.hstack((X_test_tfidf.toarray(), X_test[['length']].values / X_test['length'].max()))  # Normalize length and combine

# 逻辑回归分类
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
predictions = model.predict(X_test_tfidf)
print(classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# 分析网络暴力文本
violent_texts = X_train['cleaned_content'][y_train == '1']

# 断句
sentences = [sent for text in violent_texts for sent in sent_tokenize(text)]

# 情感分析
sentiments = [TextBlob(sent).sentiment for sent in sentences]
polarity = sum([sent.polarity for sent in sentiments]) / len(sentiments)
subjectivity = sum([sent.subjectivity for sent in sentiments]) / len(sentiments)
print(f"Average Polarity: {polarity}")
print(f"Average Subjectivity: {subjectivity}")

# 重新向量化仅用于暴力文本
X_violent_tfidf = tfidf_vectorizer.transform(sentences)

# LSA主题模型
lsa = TruncatedSVD(n_components=5)
lsa.fit(X_violent_tfidf)
terms = tfidf_vectorizer.get_feature_names_out()
print("LSA主题关键词：")
for i, comp in enumerate(lsa.components_):
    sorted_terms = sorted(zip(terms, comp), key=lambda x: x[1], reverse=True)[:10]
    print(f"主题 {i+1}: {', '.join([t[0] for t in sorted_terms])}")

# K-Means聚类
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_violent_tfidf)
plt.figure()
plt.hist(clusters, bins=range(4), align='left', color='steelblue', rwidth=0.8)
plt.title('K-Means Clustering Results')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(range(3))
plt.show()

# LDA主题模型
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X_violent_tfidf)
print("LDA主题关键词：")
for i, component in enumerate(lda.components_):
    sorted_terms = sorted(zip(terms, component), key=lambda x: x[1], reverse=True)[:10]
    print(f"主题 {i+1}: {', '.join([t[0] for t in sorted_terms])}")

# 生成词云
word_counts = pd.Series(X_violent_tfidf.toarray().sum(axis=0), index=terms).sort_values(ascending=False)
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_counts)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Cyberbullying Texts')
plt.show()

# 情感极性和主观性图
plt.figure(figsize=(6, 4))
plt.scatter([sent.polarity for sent in sentiments], [sent.subjectivity for sent in sentiments], color='blue')
plt.title('Sentiment Analysis of Cyberbullying Texts')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.grid(True)
plt.show()
