import paddle

paddle.set_device(paddle.get_device())
from paddle.io import DataLoader
from loadModel import load_model
from loadData import PoemData
from processData import data_preprocess
from paddlenlp.transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
poems = data_preprocess()
model = load_model()
train_loader = DataLoader(PoemData(poems, bert_tokenizer, 128), batch_size=8, shuffle=True)
model.fit(train_data=train_loader, epochs=10, save_dir='./checkpoint', save_freq=1, verbose=1)
