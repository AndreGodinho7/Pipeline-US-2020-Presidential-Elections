import torch
from torch import nn, optim

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

PRE_TRAINED_MODEL_NAME = 'bert-base-cased' # cased letters are important for Sentiment analysis

class BERT(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        print(attention_mask)

        _, pooled_output = self.bert(
            input_ids=input_ids, # indices of input sequence tokens in the vocabulary
            attention_mask=attention_mask # mask to avoid performing attention on padding token indices
        )
        output = self.dropout(pooled_output)
        return self.out(output)