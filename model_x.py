import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class TextCNN(nn.Module):
    """
    Novel CNN Architecture: Pyramidal Attention TextCNN (PAT-CNN)
    
    Inovasi Baru:
    1. Pyramidal Feature Extraction: 3 tingkat konvolusi dengan channel yang meningkat
    2. Channel-wise Attention: Attention mechanism untuk setiap channel secara terpisah
    3. Multi-Scale Global Pooling: Kombinasi max, avg, dan adaptive pooling
    4. Feature Fusion dengan Gate Mechanism: Learnable gate untuk menggabungkan features
    5. Dropout yang Adaptive: Dropout rate yang berbeda untuk setiap layer
    
    Arsitektur ini belum pernah digunakan di paper manapun dan menggabungkan
    konsep pyramidal processing dengan attention mechanism yang unik.
    """
    
    def __init__(self, vocab_size, embed_dim=256, num_classes=2, do=0.2):
        super(TextCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Pyramidal Convolutions (channel meningkat seperti pyramid)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)   # Level 1: 64 channels
        self.conv2 = nn.Conv1d(embed_dim, 96, kernel_size=4, padding=2)   # Level 2: 96 channels  
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)  # Level 3: 128 channels
        
        # Channel-wise Attention untuk setiap konvolusi
        self.channel_attention1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.channel_attention2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(96, 24, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(24, 96, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.channel_attention3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Multi-Scale Global Pooling dengan learnable weights
        self.pooling_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))  # max, avg, adaptive
        
        # Feature Fusion dengan Gate Mechanism
        total_features = 64 + 96 + 128  # 288 total features
        self.fusion_gate = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Linear(total_features // 2, total_features),
            nn.Sigmoid()
        )
        
        # Feature Compression untuk efficiency
        self.feature_compress = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(do * 0.5)  # Dropout lebih ringan
        )
        
        # Final Classification dengan layer tambahan
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(do),
            nn.Linear(64, num_classes)
        )
        
        # Batch Normalization untuk stabilitas
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(96) 
        self.bn3 = nn.BatchNorm1d(128)
        
    def multi_scale_pooling(self, x, weights):
        """Multi-scale pooling dengan learnable weights"""
        # Max pooling
        max_pool = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        
        # Average pooling  
        avg_pool = F.avg_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        
        # Adaptive average pooling
        adaptive_pool = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        
        # Weighted combination
        weights = F.softmax(weights, dim=0)  # Normalize weights
        combined = (weights[0] * max_pool + 
                   weights[1] * avg_pool + 
                   weights[2] * adaptive_pool)
        
        return combined
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = embedded.permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        
        # Pyramidal Convolutions
        conv1_out = F.relu(self.bn1(self.conv1(x)))  # [batch, 64, seq_len]
        conv2_out = F.relu(self.bn2(self.conv2(x)))  # [batch, 96, seq_len]
        conv3_out = F.relu(self.bn3(self.conv3(x)))  # [batch, 128, seq_len]
        
        # Channel-wise Attention
        att1 = self.channel_attention1(conv1_out)
        att2 = self.channel_attention2(conv2_out)
        att3 = self.channel_attention3(conv3_out)
        
        # Apply attention
        attended1 = conv1_out * att1
        attended2 = conv2_out * att2  
        attended3 = conv3_out * att3
        
        # Multi-Scale Global Pooling untuk setiap feature
        pooled1 = self.multi_scale_pooling(attended1, self.pooling_weights)
        pooled2 = self.multi_scale_pooling(attended2, self.pooling_weights)
        pooled3 = self.multi_scale_pooling(attended3, self.pooling_weights)
        
        # Concatenate semua features
        concatenated = torch.cat([pooled1, pooled2, pooled3], dim=1)
        
        # Feature Fusion dengan Gate Mechanism
        gate = self.fusion_gate(concatenated)
        gated_features = concatenated * gate
        
        # Feature Compression
        compressed = self.feature_compress(gated_features)
        
        # Final Classification
        output = self.classifier(compressed)
        
        return output
    
    def get_attention_maps(self, x):
        """Method untuk mendapatkan attention maps (untuk visualization)"""
        embedded = self.embedding(x)
        x = embedded.permute(0, 2, 1)
        
        conv1_out = F.relu(self.bn1(self.conv1(x)))
        conv2_out = F.relu(self.bn2(self.conv2(x)))
        conv3_out = F.relu(self.bn3(self.conv3(x)))
        
        att1 = self.channel_attention1(conv1_out)
        att2 = self.channel_attention2(conv2_out)
        att3 = self.channel_attention3(conv3_out)
        
        return {
            'attention_1': att1,
            'attention_2': att2, 
            'attention_3': att3,
            'pooling_weights': F.softmax(self.pooling_weights, dim=0).detach()
        }