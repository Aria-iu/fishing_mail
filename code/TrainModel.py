from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import os

# 邮件内容清洗
def clean_email(content):

    import jieba
    jieba.setLogLevel(0)
    # 内容分词
    content = jieba.lcut(content)

    return ' '.join(content)

def train(emails, labels):

    # 训练特征提取器
    stopwords = [word.strip() for word in open('data/stopwords.txt', encoding='utf-8', errors='ignore')]
    extractor = CountVectorizer(stop_words=stopwords, max_features=10000)
    emails = extractor.fit_transform(emails)
    features = extractor.get_feature_names_out()
    # print('数据集特征:', len(features), features)
    print('数据集大小:', len(features))

    # 实例化算法模型
    estimator = MultinomialNB(alpha=0.1)
    estimator.fit(emails, labels)

    y_preds = estimator.predict(emails)
    print('训练集准确率:', accuracy_score(labels, y_preds))

    # 指定文件夹路径
    folder_path = "model"
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created.")
    else:
        print(f"Folder {folder_path} already exists.")

    # 存储特征提取器和模型
    pickle.dump(extractor, open('model/extractor.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(estimator, open('model/estimator.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def evaluate(emails, labels):

    # 加载特征提取器
    extractor = pickle.load(open('model/extractor.pkl', 'rb'))
    # 加载算法模型
    estimator = pickle.load(open('model/estimator.pkl', 'rb'))

    # 提取特征
    emails = extractor.transform(emails)
    # 模型预测
    y_preds = estimator.predict(emails)
    print(y_preds)
    print('验证集准确率:', accuracy_score(labels, y_preds))

def check(emails):
    
    # 加载特征提取器
    extractor = pickle.load(open('model/extractor.pkl', 'rb'))
    # 加载算法模型
    estimator = pickle.load(open('model/estimator.pkl', 'rb'))

    # 提取特征
    emails = extractor.transform(emails)
    # 模型预测
    y_preds = estimator.predict(emails)

    return y_preds


if __name__ == '__main__':

    # 加载训练数据
    train_data = pickle.load(open('temp/清洗训练集.pkl', 'rb'))
    emails = train_data['email']
    labels = train_data['label']
    train(emails, labels)


    # 加载测试集数据
    test_data = pickle.load(open('temp/清洗测试集.pkl', 'rb'))
    emails = test_data['email']
    labels = test_data['label']
    evaluate(emails, labels)