import warnings

warnings.filterwarnings("ignore")
import numpy as np
import paddle
from paddleModel import PoetryBertModel
from paddlenlp.transformers import BertTokenizer
import re


class PoetryGen(object):
    """
    定义一个自动生成诗句的类，按照要求生成诗句
    model: 训练得到的预测模型
    tokenizer: 分词编码工具
    max_length: 生成诗句的最大长度，需小于等于model所允许的最大长度
    """

    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.puncs = ['，', '。', '？', '；']
        self.max_length = max_length

    def generate(self, style='', head='', topk=2):
        """
        根据要求生成诗句
        style (str): 生成诗句的风格，写成诗句的形式，如“大漠孤烟直，长河落日圆。”
        head (str, list): 生成诗句的开头内容。若head为str格式，则head为诗句开始内容；
            若head为list格式，则head中每个元素为对应位置上诗句的开始内容（即藏头诗中的头）。
        topk (int): 从预测的topk中选取结果
        """
        head_index = 0
        style_ids = self.tokenizer.encode(style)['input_ids']
        # 去掉结束标记
        style_ids = style_ids[:-1]
        head_is_list = True if isinstance(head, list) else False
        if head_is_list:
            poetry_ids = self.tokenizer.encode(head[head_index])['input_ids']
        else:
            poetry_ids = self.tokenizer.encode(head)['input_ids']
        # 去掉开始和结束标记
        poetry_ids = poetry_ids[1:-1]
        break_flag = False
        while len(style_ids) + len(poetry_ids) <= self.max_length:
            next_word = self._gen_next_word(style_ids + poetry_ids, topk)
            # 对于一些符号，如[UNK], [PAD], [CLS]等，其产生后对诗句无意义，直接跳过
            if next_word in self.tokenizer.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]']):
                continue
            if head_is_list:
                if next_word in self.tokenizer.convert_tokens_to_ids(self.puncs):
                    head_index += 1
                    if head_index < len(head):
                        new_ids = self.tokenizer.encode(head[head_index])['input_ids']
                        new_ids = [next_word] + new_ids[1:-1]
                    else:
                        new_ids = [next_word]
                        break_flag = True
                else:
                    new_ids = [next_word]
            else:
                new_ids = [next_word]
            if next_word == self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]:
                break
            poetry_ids += new_ids
            if break_flag:
                break
        return ''.join(self.tokenizer.convert_ids_to_tokens(poetry_ids))

    def _gen_next_word(self, known_ids, topk):
        type_token = [0] * len(known_ids)
        mask = [1] * len(known_ids)
        sequence_length = len(known_ids)
        known_ids = paddle.to_tensor([known_ids], dtype='int64')
        type_token = paddle.to_tensor([type_token], dtype='int64')
        mask = paddle.to_tensor([mask], dtype='float32')
        logits = self.model.network.forward(known_ids, type_token, mask, sequence_length)
        # logits中对应最后一个词的输出即为下一个词的概率
        words_prob = logits[0, -1, :].numpy()
        # 依概率倒序排列后，选取前topk个词
        words_to_be_choosen = words_prob.argsort()[::-1][:topk]
        probs_to_be_choosen = words_prob[words_to_be_choosen]
        # 归一化
        probs_to_be_choosen = probs_to_be_choosen / sum(probs_to_be_choosen)
        word_choosen = np.random.choice(words_to_be_choosen, p=probs_to_be_choosen)
        return word_choosen


def poetry_show(poetry):
    pattern = r"([，。；？])"
    text = re.sub(pattern, r'\1 ', poetry)
    for p in text.split():
        if p:
            print(p)


# 载入已经训练好的模型
net = PoetryBertModel('bert-base-chinese', 128)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = paddle.Model(net)
model.load('./model/6')
poetry_gen = PoetryGen(model, bert_tokenizer)

# 随机生成一首诗
poetry = poetry_gen.generate(head=["宫", "本", "武", "藏"])
poetry_show(poetry)
