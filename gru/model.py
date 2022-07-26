from torch import nn
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
        if attention:
            self.attention = AttentionBlock(gru_dim*2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(gru_dim*2, 1)
        )

    def forward(self, x, l):
        embedding = self.embedding(x)
        packed_input = pack_padded_sequence(embedding, l.cpu().numpy(), batch_first=True)
        packed_output, hidden_state = self.gru(packed_input)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        if self.attention is not None:
            # tbp = [(o[:5], a[:5], oa[:5]) for o, a, oa in zip(
            #     output[0],
            #     self.attention(output)[0],
            #     output[0] * self.attention(output)[0]
            # )]
            # for a, b, c in tbp:
            #     print(a, b, c)
            #     print("\n\n")
            output = output * self.attention(output)
        output = output.sum(dim=1)
        y_est = self.classifier(output)
        return y_est