import json
from config import getConfig
import io
import tensorflow as tf

# 加载参数配置文件
gConfig = getConfig.get_config()
conv_path = gConfig['resource_data']
vocab_inp_path = gConfig['vocab_inp_path']
vocab_tar_path = gConfig['vocab_tar_path']
vocab_inp_size = gConfig['vocab_inp_size']
vocab_tar_size = gConfig['vocab_tar_size']
seq_train = gConfig['seq_data']


def create_vocab(lang, vocab_path, vocab_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=3)
    tokenizer.fit_on_texts(lang)
    vocab = json.loads(tokenizer.to_json(ensure_ascii=False))
    vocab['index_word'] = tokenizer.index_word
    vocab['word_index'] = tokenizer.word_index
    vocab['document_count'] = tokenizer.document_count
    vocab = json.dumps(vocab, ensure_ascii=False)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write(vocab)
    f.close()
    print("字典保存在:{}".format(vocab_path))


def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    return w


lines = io.open(seq_train, encoding='utf-8').readlines()
word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines]

input_lang, target_lang = zip(*word_pairs)
create_vocab(input_lang, vocab_inp_path, vocab_inp_size)
create_vocab(target_lang, vocab_tar_path, vocab_tar_size)
