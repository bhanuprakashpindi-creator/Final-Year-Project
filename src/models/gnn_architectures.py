"""Graph Neural Network architectures for cryptocurrency AML classification.

Implements the four model families evaluated in Ferretti et al. (IEEE Access, 2025):
1. GCN+Lin (Graph Convolutional Network with linear projection)
2. GraphSAGE+Lin (Inductive neighborhood aggregation with linear projection)
3. GATv2+Lin (Dynamic Graph Attention Network with linear projection)
4. Chebyshev+Lin (ChebConv spectral polynomial convolution with linear projection)

All models support standard PyG Data inputs, decouple message-passing representations
through a final linear classifier, and provide comprehensive edge directionality handling.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, ChebConv
from torch_geometric.utils import to_undirected


class GCNPlusLin(nn.Module):
    """
    Two-layer Graph Convolutional Network followed by a decoupled Linear projection (GCN+Lin).
    
    Edge Policy:
        GCN message passing uses symmetric normalization D^(-1/2) A D^(-1/2).
        For directed transaction networks, edge symmetrization ensures bidirectional
        contextual flow (both incoming payer flow and outgoing payee flow).
    """
    def __init__(
        self,
        in_channels: int = 145,
        hidden_channels: int = 110,
        out_channels: int = 2,
        dropout: float = 0.5,
        symmetrize_edges: bool = True
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.symmetrize_edges = symmetrize_edges

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.symmetrize_edges:
            edge_index = to_undirected(edge_index)

        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Decoupled Linear Classification Head
        out = self.lin(h)
        return F.log_softmax(out, dim=-1)


class GraphSAGEPlusLin(nn.Module):
    """
    Two-layer GraphSAGE (Hamilton et al., 2017) with mean pooling followed by Linear projection.
    
    Edge Policy:
        By default, uses symmetrized edges for full 2-hop neighborhood aggregation.
        Optionally supports directed payment-flow aggregation if symmetrize_edges=False.
    """
    def __init__(
        self,
        in_channels: int = 145,
        hidden_channels: int = 110,
        out_channels: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
        symmetrize_edges: bool = True
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.symmetrize_edges = symmetrize_edges

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.symmetrize_edges:
            edge_index = to_undirected(edge_index)

        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Classification Head
        out = self.lin(h)
        return F.log_softmax(out, dim=-1)


class GATv2PlusLin(nn.Module):
    """
    Two-layer Dynamic Graph Attention Network (Brody et al., 2022) with Linear projection.
    
    Solves the static attention problem in standard GAT by applying dynamic multi-head
    attention coefficients over transaction neighbors.
    
    Edge Policy:
        Symmetrizes edges to allow transactions to attend to both fund sources and fund sinks.
    """
    def __init__(
        self,
        in_channels: int = 145,
        hidden_channels: int = 110,
        out_channels: int = 2,
        heads: int = 2,
        dropout: float = 0.5,
        symmetrize_edges: bool = True
    ):
        super().__init__()
        # In Layer 1: hidden_channels // heads per head so total concatenated channels = hidden_channels
        per_head_dim = max(1, hidden_channels // heads)
        concat_dim = per_head_dim * heads
        self.conv1 = GATv2Conv(in_channels, per_head_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(concat_dim, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.symmetrize_edges = symmetrize_edges

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.symmetrize_edges:
            edge_index = to_undirected(edge_index)

        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Decoupled Linear Head
        out = self.lin(h)
        return F.log_softmax(out, dim=-1)


class ChebyshevPlusLin(nn.Module):
    """
    Chebyshev Spectral Graph Convolution with Linear projection (Chebyshev+Lin).
    
    Reference Architecture from Ferretti et al. (IEEE Access, 2025, Section IV-B & Table 3).
    Approximates spectral filters using truncated Chebyshev polynomials of order K,
    granting a direct K-hop receptive field without expensive Laplacian eigendecomposition.
    
    Edge Policy:
        STRICT REQUIREMENT: Spectral graph convolutions require a symmetric graph Laplacian.
        Edge indices are unconditionally symmetrized via to_undirected(edge_index).
    """
    def __init__(
        self,
        in_channels: int = 145,
        hidden_channels: int = 110,
        out_channels: int = 2,
        K: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        self.K = K
        # Layer 1: ChebConv(in -> hidden)
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        # Layer 2: ChebConv(hidden -> hidden)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=K)
        # Final Linear classification layer
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Unconditionally enforce undirected edge index for spectral Laplacian validity
        edge_index = to_undirected(edge_index)

        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Final Linear projection + LogSoftmax
        out = self.lin(h)
        return F.log_softmax(out, dim=-1)


def build_model(
    model_name: str,
    in_channels: int = 145,
    hidden_channels: int = 110,
    out_channels: int = 2,
    dropout: float = 0.5,
    K: int = 3,
    heads: int = 2,
    symmetrize_edges: bool = True
) -> nn.Module:
    """Factory function for creating GNN model instances by name."""
    name = model_name.strip().lower().replace("_", "").replace("-", "").replace("+", "")
    
    if name in ["gcn", "gcnlin", "gcnpluslin"]:
        return GCNPlusLin(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            symmetrize_edges=symmetrize_edges
        )
    elif name in ["sage", "graphsage", "sagelin", "graphsagepluslin"]:
        return GraphSAGEPlusLin(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            symmetrize_edges=symmetrize_edges
        )
    elif name in ["gat", "gatv2", "gatv2lin", "gatv2pluslin"]:
        return GATv2PlusLin(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            symmetrize_edges=symmetrize_edges
        )
    elif name in ["cheb", "chebyshev", "chebconv", "chebyshevlin", "chebyshevpluslin"]:
        return ChebyshevPlusLin(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            K=K,
            dropout=dropout
        )
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Supported options: ['gcn', 'graphsage', 'gatv2', 'chebyshev_lin']"
        )

