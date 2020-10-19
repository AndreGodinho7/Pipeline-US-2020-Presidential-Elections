import torch
from torch import nn, optim

from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig,  AdamW, get_linear_schedule_with_warmup
PRE_TRAINED_MODEL_NAME = 'distilbert-base-cased' # cased letters are important for Sentiment analysis
TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


class DistillBERT(nn.Module):
    def __init__(self, n_classes, loading=False, dropout=0.3):
        super(DistillBERT, self).__init__()
        # config = DistilBertConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
        # self.distillbert = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config=config)
        self.distillbert = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.distillbert.resize_token_embeddings(len(TOKENIZER))
        self.drop = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(768, n_classes)
    
    def forward(self, ids, mask):
        distilbert_output = self.distillbert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output = self.out(output_1)
        return output