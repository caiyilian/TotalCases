from paddlenlp.transformers import BertModel, BertForTokenClassification
from paddle.nn import Layer
import paddle


class PoetryBertModel(Layer):
    """
    基于BERT预训练模型的诗歌生成模型
    """

    def __init__(self, pretrained_bert_model: str, input_length: int):
        super(PoetryBertModel, self).__init__()
        bert_model = BertModel.from_pretrained(pretrained_bert_model)
        self.vocab_size, self.hidden_size = bert_model.embeddings.word_embeddings.parameters()[0].shape
        self.bert_for_class = BertForTokenClassification(bert_model, self.vocab_size)
        # 生成下三角矩阵，用来mask句子后边的信息
        self.sequence_length = input_length
        self.lower_triangle_mask = paddle.tril(paddle.tensor.full((input_length, input_length), 1, 'float32'))

    def forward(self, token, token_type, input_mask, input_length=None):
        # 计算attention mask
        mask_left = paddle.reshape(input_mask, input_mask.shape + [1])
        mask_right = paddle.reshape(input_mask, [input_mask.shape[0], 1, input_mask.shape[1]])
        # 输入句子中有效的位置
        mask_left = paddle.cast(mask_left, 'float32')
        mask_right = paddle.cast(mask_right, 'float32')
        attention_mask = paddle.matmul(mask_left, mask_right)
        # 注意力机制计算中有效的位置
        if input_length is not None:
            lower_triangle_mask = paddle.tril(paddle.tensor.full((input_length, input_length), 1, 'float32'))
        else:
            lower_triangle_mask = self.lower_triangle_mask
        attention_mask = attention_mask * lower_triangle_mask
        # 无效的位置设为极小值
        attention_mask = (1 - paddle.unsqueeze(attention_mask, axis=[1])) * -1e10
        attention_mask = paddle.cast(attention_mask, self.bert_for_class.parameters()[0].dtype)

        output_logits = self.bert_for_class(token, token_type_ids=token_type, attention_mask=attention_mask)

        return output_logits


class PoetryBertModelLossCriterion(Layer):
    def forward(self, pred_logits, label, input_mask):
        loss = paddle.nn.functional.cross_entropy(pred_logits, label, ignore_index=0, reduction='none')
        masked_loss = paddle.mean(loss * input_mask, axis=0)
        return paddle.sum(masked_loss)
