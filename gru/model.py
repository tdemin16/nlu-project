from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from settings import MODEL_SIZE, PAD_TOKEN

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),

            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.attention(x)


class GRUAttention(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=MODEL_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.attention = AttentionBlock(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),

            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, l):
        embedding = self.embedding(x)
        packed_input = pack_padded_sequence(embedding, l.cpu().numpy(), batch_first=True)
        packed_output, hidden_state = self.gru(packed_input)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # context_vector = (self.attention(output) * output).sum(dim=1)
        y_est = self.classifier(output[:, -1, :])
        return y_est