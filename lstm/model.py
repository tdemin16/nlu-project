import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from settings import ATTENTION, EMBEDDING_SIZE, LSTM_SIZE, PAD_TOKEN

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.attention(x)


class LSTMAttention(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=EMBEDDING_SIZE, lstm_dim=LSTM_SIZE, attention=ATTENTION):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=PAD_TOKEN)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_dim, batch_first=True, num_layers=2, bidirectional=True)
        self.attention = None
        if attention:
            self.attention = AttentionBlock(lstm_dim*2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_dim*2, 1)
        )

    def forward(self, x, l):
        embedding = self.embedding(x)
        packed_input = pack_padded_sequence(embedding, l.cpu().numpy(), batch_first=True)

        packed_output, hidden_state = self.lstm(packed_input)
        alpha, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        if self.attention is not None:
            alpha = alpha * self.attention(alpha)
        context_vector = alpha.sum(dim=1)

        y_est = self.classifier(context_vector)
        return y_est