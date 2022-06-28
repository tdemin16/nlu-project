from torch import nn

from settings import DEVICE

class GRUAttention(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x)
        x = nn.utils.rnn.pad_sequence(x, batch_first=True).to(DEVICE)
        x = self.embedding(x)
        output, hidden_state = self.gru(x)
        x = self.linear(hidden_state)
        return self.sigmoid(x)