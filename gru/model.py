from torch import nn

from settings import MODEL_SIZE

class GRUAttention(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=MODEL_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        #self.linear = nn.Linear(embedding_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        output, hidden_state = self.gru(x)
        #print(hidden_state)
        x = hidden_state.squeeze(0)
        x = self.classifier(x) #self.linear(x)
        return self.sigmoid(x)