import os
import json
import re
from cht2chs.langconv import cht_to_chs
from paddlenlp.transformers import BertTokenizer


def sentenceParse(para):
    """
    剔除诗歌字符中的非文字符号以及数字
    """
    result, number = re.subn(u"（.*）", "", para)
    result, number = re.subn(u"{.*}", "", result)
    result, number = re.subn(u"《.*》", "", result)
    result, number = re.subn(u"《.*》", "", result)
    result, number = re.subn(u"[\]\[]", "", result)
    r = ""
    for s in result:
        if s not in set('0123456789-'):
            r += s
    r, number = re.subn(u"。。", u"。", r)
    return r


def data_preprocess(poem_dir='./chinese-poetry/json', len_limit=120):
    """
    预处理诗歌数据，返回符合要求的诗歌列表
    """
    x = 0
    poems = []
    for f in os.listdir(poem_dir):
        if f.endswith('.json'):
            json_data = json.load(open(os.path.join(poem_dir, f), encoding='utf-8'))
            for d in json_data:
                try:
                    poem = ''.join(d['paragraphs'])
                    poem = sentenceParse(poem)
                    # 控制长度，并将繁体字转换为简体字
                    if len(poem) <= len_limit:
                        poems.append(cht_to_chs(poem))
                except:
                    continue
    return poems


if __name__ == '__main__':
    # 开始处理
    poems = data_preprocess()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # 处理效果展示
    for poem in poems[6:8]:
        token_poem, _ = bert_tokenizer.encode(poem).values()
        print(poem)
        print(token_poem)
        print(''.join(bert_tokenizer.convert_ids_to_tokens(token_poem)))
