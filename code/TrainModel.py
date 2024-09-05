from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class CompositeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classifiers = [MultinomialNB(), LogisticRegression(max_iter=1000)]

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # 必须设置`classes_`属性来符合`ClassifierMixin`的规范
        for clf in self.classifiers:
            clf.fit(X, y)

    def predict(self, X):
        # 使用每个分类器进行预测，然后取平均
        predictions = [clf.predict_proba(X) for clf in self.classifiers]
        # 将概率矩阵平均
        avg_predictions = np.mean(predictions, axis=0)
        # 返回类别预测
        return self.classes_[np.argmax(avg_predictions, axis=1)]

# 邮件内容清洗
def clean_email(content):

    import jieba
    jieba.setLogLevel(0)
    # 内容分词
    content = jieba.lcut(content)

    return ' '.join(content)

def train(emails, labels):
    print('train: ')

    # 训练特征提取器
    stopwords = [word.strip() for word in open('data/stopwords.txt', encoding='utf-8', errors='ignore')]
    extractor = CountVectorizer(stop_words=stopwords, max_features=10000)
    emails = extractor.fit_transform(emails)
    features = extractor.get_feature_names_out()
    # print('数据集特征:', len(features), features)
    print('数据集特征大小:', len(features))
    # emails = [0 if feature == 'spam' else 1 for feature in features]

    # 训练Bayes模型
    estimator_bayes = MultinomialNB(alpha=0.1)
    estimator_bayes.fit(emails, labels)
    y_preds_bayes = estimator_bayes.predict(emails)
    print('Bayes模型训练完成')

    # 训练Logic模型
    estimator_logic = LogisticRegression(max_iter=1000)
    estimator_logic.fit(emails, labels)
    y_preds_logic = estimator_logic.predict(emails)
    print('Logic模型训练完成')




    # 训练Bagging模型
    composite_clf = CompositeClassifier()
    estimator_bagging = BaggingClassifier(
        # estimator=LogisticRegression(max_iter=1000),
        estimator=composite_clf,
        n_estimators=2,  # 训练多个复合分类器
        random_state=42
    )
    estimator_bagging.fit(emails, labels)
    y_preds_bagging = estimator_bagging.predict(emails)
    print('Bagging模型训练完成')


    print('训练集准确率')
    print('bayes:\t\t', accuracy_score(labels, y_preds_bayes))
    print('logic:\t\t', accuracy_score(labels, y_preds_logic))
    print('bagging:\t', accuracy_score(labels, y_preds_bagging))

    print('训练集精确率')
    print('bayes:\t\t', precision_score(labels, y_preds_bayes, pos_label='ham'))
    print('logic:\t\t', precision_score(labels, y_preds_logic, pos_label='ham'))
    print('bagging:\t', precision_score(labels, y_preds_bagging, pos_label='ham'))

    print('训练集召回率')
    print('bayes:\t\t', recall_score(labels, y_preds_bayes, pos_label='ham'))
    print('logic:\t\t', recall_score(labels, y_preds_logic, pos_label='ham'))
    print('bagging:\t', recall_score(labels, y_preds_bagging, pos_label='ham'))

    print('训练集F1-score')
    print('bayes:\t\t', f1_score(labels, y_preds_bayes, pos_label='ham'))
    print('logic:\t\t', f1_score(labels, y_preds_logic, pos_label='ham'))
    print('bagging:\t', f1_score(labels, y_preds_bagging, pos_label='ham'))

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
    pickle.dump(estimator_bayes, open('model/estimator_bayes.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(estimator_logic, open('model/estimator_logic.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(estimator_bagging, open('model/estimator_bagging.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def evaluate(emails, labels):

    # 加载特征提取器
    extractor = pickle.load(open('model/extractor.pkl', 'rb'))
    # 加载算法模型
    estimator_bayes = pickle.load(open('model/estimator_bayes.pkl', 'rb'))
    estimator_logic = pickle.load(open('model/estimator_logic.pkl', 'rb'))
    estimator_bagging = pickle.load(open('model/estimator_bagging.pkl', 'rb'))

    # 提取特征
    emails = extractor.transform(emails)

    # 模型预测
    y_preds_bayes = estimator_bayes.predict(emails)
    y_preds_logic = estimator_logic.predict(emails)
    y_preds_bagging = estimator_bagging.predict(emails)


    print('验证集准确率')
    print('bayes:\t\t', accuracy_score(labels, y_preds_bayes))
    print('logic:\t\t', accuracy_score(labels, y_preds_logic))
    print('bagging:\t', accuracy_score(labels, y_preds_bagging))

    print('验证集精确率')
    print('bayes:\t\t', precision_score(labels, y_preds_bayes, pos_label='ham'))
    print('logic:\t\t', precision_score(labels, y_preds_logic, pos_label='ham'))
    print('bagging:\t', precision_score(labels, y_preds_bagging, pos_label='ham'))

    print('验证集召回率')
    print('bayes:\t\t', recall_score(labels, y_preds_bayes, pos_label='ham'))
    print('logic:\t\t', recall_score(labels, y_preds_logic, pos_label='ham'))
    print('bagging:\t', recall_score(labels, y_preds_bagging, pos_label='ham'))

    print('验证集F1-score')
    print('bayes:\t\t', f1_score(labels, y_preds_bayes, pos_label='ham'))
    print('logic:\t\t', f1_score(labels, y_preds_logic, pos_label='ham'))
    print('bagging:\t', f1_score(labels, y_preds_bagging, pos_label='ham'))

import email
from email import policy
from email.parser import BytesParser
def read_email_file(email_file_path):
    with open(email_file_path, 'rb') as email_file:
        msg = BytesParser(policy=policy.default).parse(email_file)
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body = part.get_content()
                return body
        return None

def recognize(email_file_paths):

    emails = [read_email_file(email_file_path) for email_file_path in email_file_paths]

    # 加载特征提取器
    extractor = pickle.load(open('model/extractor.pkl', 'rb'))
    # 加载算法模型
    estimator = pickle.load(open('model/estimator_logic.pkl', 'rb'))

    # TrainModel.py
    print(emails)  # 添加这一行来输出 emails 列表，查看其内容
    emails = extractor.transform(emails)
    # 模型预测
    y_preds = estimator.predict(emails)
    results = ['正常邮件' if predict == 'ham' else '钓鱼邮件' for predict in y_preds]

    return results


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

    print(recognize(['test.eml']))