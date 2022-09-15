import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from cvtDict import data_name, model_name
from visualization import tree_vis, studyVis
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']


class Model(object):
    def __init__(self, name: str):
        self.model_name = name
        self.model = self.load(name)

    def load(self, name: str):
        if name == model_name['决策树']:
            try:
                model = DecisionTreeClassifier()
            except NameError:
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier()
        elif name == model_name['随机森林']:
            try:
                model = RandomForestClassifier(n_estimators=10, oob_score=True)
            except NameError:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=10, oob_score=True)

        elif name == model_name['支持向量机']:
            try:
                model = SVC()
            except NameError:
                from sklearn.svm import SVC
                model = SVC()
        elif name == model_name['神经网络']:
            try:
                model = BPnet()
            except NameError:
                from network import BPnet
                model = BPnet()
        else:
            raise ValueError("没有这个模型")
        return model

    def train(self, x_train, y_train, visible=False, dataName=''):
        scores = []
        if self.model_name == model_name["神经网络"]:
            scores = self.model.fit(x_train, y_train)
        else:
            for i in range(math.ceil(len(y_train) / 15)):
                try:
                    self.model.fit(x_train[:(i + 1) * 10], y_train[:(i + 1) * 10])
                except:
                    self.model.fit(x_train, y_train)
                scores.append(self.model.score(x_train, y_train))
        if visible:
            studyVis(scores)
            if self.model_name == model_name['决策树']:
                tree_vis(self.model, dataName)

            elif self.model_name == model_name["随机森林"]:
                tree_vis(self.model.estimators_[0], dataName)




class Datasets(object):
    def __init__(self, name):
        self.data_name = name
        self.data = self.load(name)
        self.__train_rate = 0.75
        self.split()

    def load(self, name):
        if name == data_name['鸢尾花']:
            try:
                data = load_iris()
            except NameError:
                from sklearn.datasets import load_iris
                data = load_iris()
        elif name == data_name['手写数字']:
            try:
                data = load_digits()
            except NameError:
                from sklearn.datasets import load_digits
                data = load_digits()
        elif name == data_name['红酒']:
            try:
                data = load_wine()
            except NameError:
                from sklearn.datasets import load_wine
                data = load_wine()
        elif name == data_name['威斯康辛州乳腺癌']:
            try:
                data = load_breast_cancer()
            except NameError:
                from sklearn.datasets import load_breast_cancer
                data = load_breast_cancer()
        else:
            raise ValueError("没有这个数据集")
        return data

    def split(self):
        X, Y = self.data.data, self.data.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, train_size=self.__train_rate)

    def setSplitNum(self, train_rate):
        self.__train_rate = train_rate
        self.split()

if __name__ == '__main__':
    dataName = data_name["威斯康辛州乳腺癌"]
    modelName = model_name["神经网络"]
    model = Model(modelName)
    dataset = Datasets(dataName)
    model.train(dataset.x_train, dataset.y_train, visible=True, dataName=dataName)
    print(model.model.score(dataset.x_test, dataset.y_test))
