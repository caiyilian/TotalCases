import numpy as np
from sklearn.preprocessing import normalize


def sigmod(z):
    h = 1. / (1 + np.exp(-z))
    return h


def de_sigmoid(z, h):
    return h * (1 - h)


def relu(z):
    h = np.maximum(z, 0)
    return h


def de_relu(z, h):
    z[z <= 0] = 0
    z[z > 0] = 1.0
    return z


def no_active(z):
    h = z
    return h


def de_no_active(z, h):
    return np.ones(h.shape)


# o Nxc
# lab Nxc
def loss_L2(o, lab):
    diff = lab - o
    sqrDiff = diff ** 2
    return 0.5 * np.sum(sqrDiff)


def de_loss_L2(o, lab):
    return o - lab


def loss_CE(o, lab):
    p = np.exp(o) / np.sum(np.exp(o), axis=1, keepdims=True)
    loss_ce = np.sum(-lab * np.log(p))
    return loss_ce


def de_loss_CE(o, lab):
    p = np.exp(o) / np.sum(np.exp(o), axis=1, keepdims=True)
    return p - lab


# dim_in:输入特征的维度
# list_num_hidden： 每层输出节点的数目
# list_act_funs： 每层的激活函数
# list_de_act_funs: 反向传播时的函数

def bulid_net(dim_in, list_num_hidden,
              list_act_funs, list_de_act_funs):
    layers = []

    # 逐层的进行网络构建
    for i in range(len(list_num_hidden)):
        layer = {}

        # 定义每一层的权重
        if i == 0:
            # layer["w"]= 0.2*np.random.randn(dim_in,list_num_hidden[i])-0.1 # 用sigmoid激活函数
            layer["w"] = 0.01 * np.random.randn(dim_in, list_num_hidden[i])  # 用relu 激活函数
        else:
            # layer["w"]= 0.2*np.random.randn(list_num_hidden[i-1],list_num_hidden[i])-0.1 # 用sigmoid激活函数
            layer["w"] = 0.01 * np.random.randn(list_num_hidden[i - 1], list_num_hidden[i])  # 用relu 激活函数

        # 定义每一层的偏置
        layer["b"] = 0.1 * np.ones([1, list_num_hidden[i]])
        layer["act_fun"] = list_act_funs[i]
        layer["de_act_fun"] = list_de_act_funs[i]
        layers.append(layer)

    return layers


# 返回每一层的输入
# 与最后一层的输出
def fead_forward(datas, layers):
    input_layers = []
    input_acfun = []
    for i in range(len(layers)):
        layer = layers[i]
        if i == 0:
            inputs = datas
            z = np.dot(inputs, layer["w"]) + layer["b"]
            h = layer['act_fun'](z)
            input_layers.append(inputs)
            input_acfun.append(z)
        else:
            inputs = h
            z = np.dot(inputs, layer["w"]) + layer["b"]
            h = layer['act_fun'](z)
            input_layers.append(inputs)
            input_acfun.append(z)
    return input_layers, input_acfun, h


# 进行参数更新更新
def updata_wb(datas, labs, layers, loss_fun, de_loss_fun, alpha=0.01):
    N, D = np.shape(datas)
    # 进行前馈操作
    inputs, input_acfun, output = fead_forward(datas, layers)
    # 计算 loss
    loss = loss_fun(output, labs)
    # 从后向前计算
    deltas0 = de_loss_fun(output, labs)
    # 从后向前计算误差
    deltas = []
    for i in range(len(layers)):
        index = -i - 1
        if i == 0:
            h = output
            z = input_acfun[index]
            delta = deltas0 * layers[index]["de_act_fun"](z, h)

        else:
            h = inputs[index + 1]
            z = input_acfun[index]
            delta = np.dot(delta, layers[index + 1]["w"].T) * layers[index]["de_act_fun"](z, h)

        deltas.insert(0, delta)

    # 利用误差 对每一层的权重进行修成
    for i in range(len(layers)):
        # 计算 dw 与 db
        dw = np.dot(inputs[i].T, deltas[i])

        db = np.sum(deltas[i], axis=0, keepdims=True)
        # 梯度下降
        layers[i]["w"] = layers[i]["w"] - alpha * dw
        # print(alpha * dw)
        # print("-" * 10)
        layers[i]["b"] = layers[i]["b"] - alpha * db
    return layers, loss


def test_accuracy(datas, labs_true, layers):
    _, _, output = fead_forward(datas, layers)
    lab_det = np.argmax(output, axis=1)
    labs_true = np.argmax(labs_true, axis=1)
    N_error = np.where(np.abs(labs_true - lab_det) > 0)[0].shape[0]

    error_rate = N_error / np.shape(datas)[0]
    return error_rate


def load_dataset_iris(file_data, N_train):
    # 数据读取
    datas = np.loadtxt(file_data, dtype=np.float, delimiter=',', usecols=(0, 1, 2, 3))
    labs = np.loadtxt(file_data, dtype=str, delimiter=',', usecols=(4))
    N, D = np.shape(datas)
    N_test = N - N_train
    unqiue_labs = np.unique(labs).tolist()

    dic_str2index = {}
    dic_index2str = {}
    for i in range(len(unqiue_labs)):
        lab_str = unqiue_labs[i]
        dic_str2index[lab_str] = i
        dic_index2str[i] = lab_str

    labs_onehot = np.zeros([N, len(unqiue_labs)])
    for i in range(N):
        labs_onehot[i, dic_str2index[labs[i]]] = 1

    perm = np.random.permutation(N)
    index_train = perm[:N_train]
    index_test = perm[N_train:]

    data_train = datas[index_train, :]
    lab_train_onehot = labs_onehot[index_train, :]

    data_test = datas[index_test, :]
    lab_test_onehot = labs_onehot[index_test]

    return data_train, lab_train_onehot, data_test, lab_test_onehot, dic_index2str


def one_hot(target, classNum):
    N = len(target)
    lab_onehot = np.zeros([N, classNum])
    for i in range(N):
        id = int(target[i])
        lab_onehot[i, id] = 1
    return lab_onehot


class BPnet():
    def __init__(self):
        pass

    def __initNetStruct(self):
        list_num_hidden = [30, 5, self.classNum]
        list_act_funs = [relu, relu, no_active]
        list_de_act_funs = [de_relu, de_relu, de_no_active]
        # 定义损失函数
        self.loss_fun = loss_CE
        self.de_loss_fun = de_loss_CE
        self.model = bulid_net(self.featureNum, list_num_hidden,
                               list_act_funs, list_de_act_funs)

    def fit(self, x_train, y_train):
        self.train_data = normalize(x_train, axis=0, norm='max')
        self.classNum = np.max(y_train) + 1
        self.featureNum = x_train.shape[1]
        self.train_lab_onehot = one_hot(y_train, self.classNum)
        self.__initNetStruct()
        scores = []
        # 进行训练
        n_epoch = 5000

        batchsize = 20
        N = x_train.shape[0]
        N_batch = N // batchsize
        for i in range(n_epoch):
            # 数据打乱
            rand_index = np.random.permutation(N).tolist()
            # 每个batch 更新一下weight
            loss_sum = 0
            for j in range(N_batch):
                index = rand_index[j * batchsize:(j + 1) * batchsize]
                batch_datas = self.train_data[index]
                batch_labs = self.train_lab_onehot[index]
                layers, loss = updata_wb(batch_datas, batch_labs, self.model, self.loss_fun, self.de_loss_fun,
                                         alpha=0.01)
                loss_sum = loss_sum + loss

            error = test_accuracy(self.train_data, self.train_lab_onehot, self.model)
            score = 1 - error
            interval = n_epoch // 100
            if i % interval == 0:
                scores.append(score * 100)
        return scores

    def score(self, x_test, y_test):
        x_test = normalize(x_test, axis=0, norm='max')
        test_lab_onehot = one_hot(y_test, self.classNum)
        error = test_accuracy(x_test, test_lab_onehot, self.model)
        return (1 - error) * 100

