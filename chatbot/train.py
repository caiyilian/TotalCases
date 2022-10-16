# 导入依赖包
import json
import os
import sys
import time
import tensorflow as tf
import seq2seqModel
from config import getConfig
import io

# 初始化超参字典，并对相应的参数进行赋值
gConfig = getConfig.get_config()
vocab_inp_size = gConfig['vocab_inp_size']
vocab_tar_size = gConfig['vocab_tar_size']
embedding_dim = gConfig['embedding_dim']
units = gConfig['layer_size']
BATCH_SIZE = gConfig['batch_size']

max_length_inp = gConfig['max_length']
max_length_tar = gConfig['max_length']

log_dir = gConfig['log_dir']
writer = tf.summary.create_file_writer(log_dir)


# 对训练语料进行处理，上下文分别加上start end标示
def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    return w


# 定义数据读取函数，从训练语料中读取数据并进行word2number的处理，并生成词典
def read_data(path):
    path = os.getcwd() + '/' + path
    if not os.path.exists(path):
        path = os.path.dirname(os.getcwd()) + '/' + path
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines]
    input_lang, target_lang = zip(*word_pairs)
    input_tokenizer = tokenize(gConfig['vocab_inp_path'])
    target_tokenizer = tokenize(gConfig['vocab_tar_path'])
    input_tensor = input_tokenizer.texts_to_sequences(input_lang)
    target_tensor = target_tokenizer.texts_to_sequences(target_lang)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length_inp,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length_tar,
                                                                  padding='post')
    return input_tensor, input_tokenizer, target_tensor, target_tokenizer


# 定义word2number函数，通过对语料的处理提取词典，并进行word2number处理以及padding补全
def tokenize(vocab_file):
    # 从词典中读取预先生成tokenizer的config，构建词典矩阵
    with open(vocab_file, 'r', encoding='utf-8') as f:
        tokenize_config = json.dumps(json.load(f), ensure_ascii=False)
        lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenize_config)
    # 利用词典进行word2number的转换以及padding处理
    return lang_tokenizer


input_tensor, input_token, target_tensor, target_token = read_data(gConfig['seq_data'])

steps_per_epoch = len(input_tensor) // (gConfig['batch_size'])
BUFFER_SIZE = len(input_tensor)
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
enc_hidden = seq2seqModel.encoder.initialize_hidden_state()


# 定义训练函数
def train():
    # 从训练语料中读取数据并使用预生成词典word2number的转换
    print("Preparing data in %s" % gConfig['train_data'])
    print('每个epoch的训练步数: {}'.format(steps_per_epoch))
    # 如有已经有预训练的模型则加载预训练模型继续训练
    checkpoint_dir = gConfig['model_data']
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        print("reload pretrained model")
        seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # 使用Dataset加载训练数据，Dataset可以加速数据的并发读取并进行训练效率的优化
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    start_time = time.time()
    # current_loss=2
    # min_loss=gConfig['min_loss']
    epoch = 0
    train_epoch = gConfig['train_epoch']
    # 开始进行循环训练，这里设置了一个结束循环的条件就是当loss小于设置的min_loss超参时终止训练
    while epoch < train_epoch:
        start_time_epoch = time.time()
        total_loss = 0
        # 进行一个epoch的训练，训练的步数为steps_per_epoch
        for batch, (inp, targ) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = seq2seqModel.training_step(inp, targ, target_token, enc_hidden, batch == 0)
            total_loss += batch_loss
            print('epoch:{}batch:{} batch_loss: {}'.format(epoch, batch, batch_loss))
        # 结束一个epoch的训练后，更新current_loss，计算在本epoch中每步训练平均耗时、loss值
        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        epoch_time_total = (time.time() - start_time)
        print('训练总步数: {} 总耗时: {}  epoch平均每步耗时: {} 平均每步loss {:.4f}'
              .format(current_steps, epoch_time_total, step_time_epoch, step_loss))
        # 将本epoch训练的模型进行保存，更新模型文件
        seq2seqModel.checkpoint.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()
        epoch = epoch + 1
        with writer.as_default():
            tf.summary.scalar('loss', step_loss, step=epoch)


# 定义预测函数，用于根据上文预测下文对话


# main函数的入口，根据超参设置的模式启动不同工作模式
if __name__ == '__main__':
    train()
