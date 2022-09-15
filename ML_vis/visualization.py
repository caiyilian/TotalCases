import matplotlib.pyplot as plt
from cvtDict import feature_names, target_names
from sklearn.tree import plot_tree
import numpy as np
from matplotlib.animation import FuncAnimation

x, y = [], []  # 用于保存绘图数据，最开始时什么都没有，默认为空


def update(score):  # 更新函数
    x.append(len(y))  # 添加X轴坐标
    y.append(score)  # 添加Y轴坐标
    plt.plot(x, y, "r--")  # 绘制折线图


def tree_vis(model, dataName):
    fn = feature_names[dataName]
    cn = target_names[dataName]
    plt.figure("决策树或随机森林的可视化")
    plot_tree(model, filled=True, feature_names=fn, class_names=cn)


def studyVis(scores):
    fig = plt.figure("学习曲线(准确率)")
    plt.ylim(min(scores) * 0.999, max(scores) * 1.01)  # Y轴取值范围
    plt.ylabel("准确率", )  # Y轴刻度
    plt.xlim(0, len(scores) + 1)  # X轴取值范围
    plt.xlabel("训练轮数")  # Y轴刻度
    global x, y
    x, y = [], []  # 用于保存绘图数据，最开始时什么都没有，默认为空
    ani = FuncAnimation(fig, update, frames=scores, interval=3000 / len(scores), blit=False, repeat=False)  # 创建动画效果
    plt.show()
