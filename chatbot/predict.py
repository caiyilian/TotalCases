
import json
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import seq2seqModel
from config import getConfig
import jieba
import tensorflow as tf

# 初始化超参字典，并对相应的参数进行赋值
gConfig = getConfig.get_config()
units = gConfig['layer_size']

max_length_inp = gConfig['max_length']
max_length_tar = gConfig['max_length']


# 对训练语料进行处理，上下文分别加上start end标示
def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    return w


# 定义word2number函数，通过对语料的处理提取词典，并进行word2number处理以及padding补全
def tokenize(vocab_file):
    # 从词典中读取预先生成tokenizer的config，构建词典矩阵
    with open(vocab_file, 'r', encoding='utf-8') as f:
        tokenize_config = json.dumps(json.load(f), ensure_ascii=False)
        lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenize_config)
    # 利用词典进行word2number的转换以及padding处理
    return lang_tokenizer


# 定义预测函数，用于根据上文预测下文对话
def predict(sentence):
    # 从词典中读取预先生成tokenizer的config，构建词典矩阵
    input_tokenizer = tokenize(gConfig['vocab_inp_path'])
    target_tokenizer = tokenize(gConfig['vocab_tar_path'])
    # 加载预训练的模型
    checkpoint_dir = gConfig['model_data']
    seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # 对输入的语句进行处理，加上start end标示
    sentence = " ".join(jieba.cut(sentence))
    sentence = (preprocess_sentence(sentence),)

    # 进行word2number的转换
    inputs = input_tokenizer.texts_to_sequences(sentence)
    inputs = inputs[0]
    # 进行padding的补全
    if None in inputs:
        return "emm...不知道该说什么"
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    # 初始化一个中间状态
    hidden = [tf.zeros((1, units))]
    # 对输入上文进行encoder编码，提取特征
    enc_out, enc_hidden = seq2seqModel.encoder(inputs, hidden)
    dec_hidden = enc_hidden
    # decoder的输入从start的对应Id开始正向输入
    dec_input = tf.expand_dims([target_tokenizer.word_index['start']], 0)
    # 在最大的语句长度范围内容，使用模型中的decoder进行循环解码
    for t in range(max_length_tar):
        # 获得解码结果，并使用argmax确定概率最大的id
        predictions, dec_hidden, attention_weights = seq2seqModel.decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        # 判断当前Id是否为语句结束表示，如果是则停止循环解码，否则进行number2word的转换，并进行语句拼接
        if target_tokenizer.index_word[predicted_id] == 'end':
            break
        result += str(target_tokenizer.index_word[predicted_id]) + ' '
        # 将预测得到的id作为下一个时刻的decoder的输入
        dec_input = tf.expand_dims([predicted_id], 0)
    return result


# main函数的入口，根据超参设置的模式启动不同工作模式
if __name__ == '__main__':
    result = predict("你好")
    print(result)