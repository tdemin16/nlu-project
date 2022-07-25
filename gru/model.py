from unicodedata import bidirectional
import torch

from torch import dropout, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from settings import ATTENTION, EMBEDDING_SIZE, GRU_SIZE, PAD_TOKEN

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.attention(x)


class GRUAttention(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=EMBEDDING_SIZE, gru_dim=GRU_SIZE, attention=ATTENTION):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_dim, batch_first=True, num_layers=2, bidirectional=True)
        self.attention = None
        self.dropout = None
        if attention:
            self.attention = AttentionBlock(gru_dim*2)
        else:
            self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(gru_dim*2, 1)

    def forward(self, x, l):
        embedding = self.embedding(x)
        packed_input = pack_padded_sequence(embedding, l.cpu().numpy(), batch_first=True)
        packed_output, hidden_state = self.gru(packed_input)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        if self.attention is not None:
            output = output * self.attention(output)
        output = output.sum(dim=1)
        if self.attention is None:
            output = self.dropout(output)
        y_est = self.linear(output)
        return y_est