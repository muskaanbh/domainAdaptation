from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F

class BERTClass(nn.Module):
  def __init__(self,n_classes,dropout_rate):
    super(BERTClass,self).__init__()
    self.bert_model = BertModel.from_pretrained("bert-base-uncased",return_dict=True)
    self.drop = nn.Dropout(p=dropout_rate)
    self.linear = nn.Linear(768,n_classes)
    # self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids,attention_mask, token_type_ids):
    output = self.bert_model(input_ids,attention_mask, token_type_ids)
    output = self.drop(output.pooler_output)
    output = self.linear(output)
    return output