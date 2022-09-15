from paddle.io import Dataset
import numpy as np


class PoemData(Dataset):
    """
    构造诗歌数据集，继承paddle.io.Dataset
    Parameters:
        poems (list): 诗歌数据列表，每一个元素为一首诗歌，诗歌未经编码
        max_len: 接收诗歌的最大长度
    """

    def __init__(self, poems, tokenizer, max_len=128):
        super(PoemData, self).__init__()
        self.poems = poems
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        line = self.poems[idx]
        token_line = self.tokenizer.encode(line)
        token, token_type = token_line['input_ids'], token_line['token_type_ids']
        if len(token) > self.max_len + 1:
            token = token[:self.max_len] + token[-1]
            token_type = token_type[:self.max_len] + token_type[-1]
        input_token, input_token_type = token[:-1], token_type[:-1]
        label_token = np.array((token[1:] + [0] * self.max_len)[:self.max_len], dtype='int64')
        # 输入填充
        input_token = np.array((input_token + [0] * self.max_len)[:self.max_len], dtype='int64')
        input_token_type = np.array((input_token_type + [0] * self.max_len)[:self.max_len], dtype='int64')
        input_pad_mask = (input_token != 0).astype('float32')
        return input_token, input_token_type, input_pad_mask, label_token, input_pad_mask

    def __len__(self):
        return len(self.poems)
