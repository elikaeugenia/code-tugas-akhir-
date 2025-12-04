import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=2, do=0.2):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Layer CNN yang digunakan:
        self.conv1 = nn.Conv1d(embed_dim, 50, kernel_size=3)  # Conv1D kernel size 3
        self.conv2 = nn.Conv1d(embed_dim, 50, kernel_size=4)  # Conv1D kernel size 4
        self.conv3 = nn.Conv1d(embed_dim, 50, kernel_size=5)  # Conv1D kernel size 5
        self.dropout = nn.Dropout(do)
        self.fc = nn.Linear(150, num_classes)
 
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x1 = F.relu(self.conv1(x)).max(dim=2)[0]
        x2 = F.relu(self.conv2(x)).max(dim=2)[0]
        x3 = F.relu(self.conv3(x)).max(dim=2)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        return self.fc(x)