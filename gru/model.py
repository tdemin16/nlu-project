import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from settings import ATTENTION, EMBEDDING_SIZE, GRU_SIZE, PAD_TOKEN

class AttentionBlock(nn.Module):
    """
    Outputs a probability distribution on dimension 1.
    """
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Dropout(0.3),
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
        # instantiate AttentionBlock only if attention==True
        # gru_dim * 2 due to bidirectional GRU
        if attention:
            self.attention = AttentionBlock(gru_dim*2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(gru_dim*2, 1)
        )

    def forward(self, x, l):
        """
        Params:
        -----
        x: input sequence with batch_first = True
        l: lengths
        """
        embedding = self.embedding(x)
        packed_input = pack_padded_sequence(embedding, l.cpu().numpy(), batch_first=True)

        packed_output, hidden_state = self.gru(packed_input)
        alpha, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # compute attention only if attention is not None
        if self.attention is not None:
            alpha = alpha * self.attention(alpha)
        
        # sum outputs of GRU either if attention is True either if it is false
        context_vector = alpha.sum(dim=1)
        y_est = self.classifier(context_vector)
        return y_est

    def get_attention_heatmap(self, x, l):
        """
        Compute just the attentions scores for each sequence without computing
        the actual output of classification. Used to compute attention
        heatmaps.

        Params:
        -----
        x: input sequence with batch_first = True
        l: lengths

        Returns:
        ------
        Probability distribution over each sequence
        """
        embedding = self.embedding(x)
        packed_input = pack_padded_sequence(embedding, l.cpu().numpy(), batch_first=True)

        packed_output, hidden_state = self.gru(packed_input)
        alpha, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        return self.attention(alpha)

        