
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class GNNBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINConv(self.mlp)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return x + h


class PaperContextGNN(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        bottleneck_dim=256,
        num_layers=3,
        dropout=0.3,
    ):
        super().__init__()

        # 1. 输入压缩
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, bottleneck_dim),
        )

        # 2. GNN core
        self.layers = nn.ModuleList([
            GNNBlock(bottleneck_dim, dropout)
            for _ in range(num_layers)
        ])

        # 3. 输出 delta
        self.delta_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, input_dim),
        )

        self.out_norm = nn.LayerNorm(input_dim)

    def forward(self, data):
        """
        data.x          : [N, 1024]
        data.edge_index : [2, E]
        """
        x0 = data.x

        x = self.in_proj(x0)
        for layer in self.layers:
            x = layer(x, data.edge_index)

        delta = self.delta_proj(x)

        # 统一 residual 更新（所有节点）
        x_out = x0 + delta

        return self.out_norm(x_out)
