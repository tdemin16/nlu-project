from torch import nn
from transformers import AutoModelForSequenceClassification

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cffl/bert-base-styleclassification-subjective-neutral", 
            num_labels=1,
            ignore_mismatched_sizes=True
        )
    
    def forward(self, x, a):
        raw_output = self.model(x, attention_mask=a).logits
        return raw_output