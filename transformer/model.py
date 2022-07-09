from torch import nn
from transformers import AutoModelForSequenceClassification

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, a):
        raw_output = self.model(x, attention_mask=a).logits
        return self.sigmoid(raw_output)