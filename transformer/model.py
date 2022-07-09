from torch import nn
from transformers import BertModel

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, a):
        output, pooled_output = self.backbone(x, attention_mask=a, return_dict=False)
        est = self.classifier(pooled_output)
        return est