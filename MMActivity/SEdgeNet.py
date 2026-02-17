import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_knn_indices(x, k):
    """
    Compute the k-nearest neighbors for each point in x.

    Args:
        x (torch.Tensor): Shape (B, C, N).
        k (int): Number of neighbors.

    Returns:
        torch.Tensor: The indices of the top-k neighbors, shape (B, N, k).
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx  # shape (B, N, k)

def get_graph_feature(x, k=10, idx=None):
    """
    Compute the graph feature for input x.

    Args:
        x (torch.Tensor): Input of shape (B, C, N).
        k (int, optional): Number of nearest neighbors. Default is 10.
        idx (torch.Tensor, optional): Precomputed adjacency indices flattened (B*N*k).
                                      If None, they will be computed and returned as flattened.

    Returns:
        tuple: (feature, idx)
            - feature (torch.Tensor): Graph feature of shape (B, 2*C, N, k').
            - idx (torch.Tensor): The adjacency indices used (flattened B*N*k').
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx_mat = compute_knn_indices(x, k=k)  # (B, N, k)
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = (idx_mat + idx_base).view(-1)  # flatten to (B*N*k,)
    else:
        # assume idx is flattened (B*N*k')
        # keep as-is; we'll later reshape to (B, N, k') in callers if needed
        pass

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (B*N*k', C)
    # determine real k' from idx length
    kprime = feature.shape[0] // (batch_size * num_points)
    feature = feature.view(batch_size, num_points, kprime, num_dims)  # (B, N, k', C)
    x_central = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, kprime, 1)
    feature = torch.cat((feature - x_central, x_central), dim=3).permute(0, 3, 1, 2).contiguous()
    # (B, 2*C, N, k')
    return feature, idx

class SamplingEdgeConv(nn.Module):
    """
    EdgeConv-like layer with:
      - per-layer sampling of s neighbors from the precomputed k neighbors
      - optional depthwise separable 1x1 conv over edge channels (2*C -> out)
    """

    def __init__(self, in_ch, out_ch, k, sample_ratio=0.5, depthwise=True, bn=True):
        """
        in_ch: channels per node (C)
        out_ch: output node channels after aggregation
        k: precomputed number of neighbors (k)
        sample_ratio: fraction of neighbors to use per forward pass (s = max(1, int(k*sample_ratio)))
        depthwise: whether to use depthwise separable conv on edge features
        """
        super().__init__()
        self.k = k
        self.sample_ratio = float(sample_ratio)
        self.depthwise = depthwise

        self.sample_k = max(1, int(math.ceil(k * self.sample_ratio)))

        # Batchnorm and activations
        self.bn2d = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.bn_dw = nn.BatchNorm2d(in_ch * 2) if bn and depthwise else (nn.BatchNorm2d(out_ch) if bn else nn.Identity())
        self.act = nn.LeakyReLU(0.2)

        if depthwise:
            # depthwise across channels (in_channels = 2*in_ch)
            self.depthwise_conv = nn.Conv2d(in_ch * 2, in_ch * 2, kernel_size=1, groups=in_ch * 2, bias=False)
            # pointwise to out_ch
            self.pointwise_conv = nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, bias=False)
        else:
            # single conv mapping (2*in_ch -> out_ch)
            self.edge_conv = nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, bias=False)

    def _sample_indices(self, common_idx_flat, batch_size, num_points):
        """
        common_idx_flat: flattened idx returned originally by get_graph_feature on points (B*N*k)
        returns: sampled_idx_flat corresponding to (B*N*sample_k)
        """
        # reshape to (B, N, k)
        idx_mat = common_idx_flat.view(batch_size, num_points, self.k)  # (B,N,k)
        # build random selection along last dim: choose sample_k positions
        # Create random scores and take top-sample_k per node (vectorized)
        device = idx_mat.device
        # random scores
        rand_scores = torch.rand(batch_size, num_points, self.k, device=device)
        topk_vals, topk_pos = rand_scores.topk(self.sample_k, dim=-1)  # (B,N,s)
        # gather indices at those positions
        # prepare gather indices to index axis -1
        # expand gather index to match dims for gather
        gathered = torch.gather(idx_mat, dim=2, index=topk_pos)  # (B,N,s)
        sampled_idx_flat = (gathered + 0).view(-1)  # flatten to (B*N*s)
        return sampled_idx_flat

    def forward(self, x, common_idx_flat):
        """
        x: (B, C, N)
        common_idx_flat: flattened indices for original k neighbors (B*N*k)
        returns: (B, out_ch, N)
        """
        B, C, N = x.shape
        # sample s neighbors per node
        sampled_idx_flat = self._sample_indices(common_idx_flat, B, N)  # (B*N*s)

        # build edge features with sampled neighbors
        edge_feat, _ = get_graph_feature(x, k=self.sample_k, idx=sampled_idx_flat)
        # edge_feat shape: (B, 2*C, N, s)

        if self.depthwise:
            # depthwise 1x1 across channels then pointwise
            dw = self.depthwise_conv(edge_feat)     # (B, 2C, N, s)
            dw = self.bn_dw(dw)
            dw = self.act(dw)
            out = self.pointwise_conv(dw)           # (B, out, N, s)
        else:
            out = self.edge_conv(edge_feat)         # (B, out, N, s)
            out = self.bn2d(out)
            out = self.act(out)

        # aggregate over sampled neighbors (max)
        out = out.max(dim=-1, keepdim=False)[0]     # (B, out, N)
        return out

class Net(nn.Module):
    def __init__(self, args, output_channels=5):
        super(Net, self).__init__()
        self.args = args
        self.k = args.k
        self.num_layers = args.num_layers  

        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        # sampling knobs (safe defaults)
        self.use_sampling = getattr(args, "use_sampling", True)
        self.sample_ratio = getattr(args, "sample_ratio", 0.5)   # fraction of neighbors to keep
        self.depthwise = getattr(args, "depthwise", True)

        # Define channel progression
        base_channels = [64, 64, 128, 256]
        self.channels = base_channels[:self.num_layers]
        if self.num_layers > len(base_channels):
            for i in range(len(base_channels), self.num_layers):
                self.channels.append(base_channels[-1] * 2 ** (i - len(base_channels) + 1))

        self.layers = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        prev_ch = 3
        for i in range(self.num_layers):
            in_ch = prev_ch
            out_ch = self.channels[i]
            if self.use_sampling:
                layer = SamplingEdgeConv(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    k=self.k,
                    sample_ratio=self.sample_ratio,
                    depthwise=self.depthwise,
                    bn=True
                )
                self.layers.append(layer)
                self.bn_list.append(nn.Identity())  
            else:
                self.layers.append(nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, bias=False))
                self.bn_list.append(nn.BatchNorm2d(out_ch))
            prev_ch = out_ch

        total_ch = sum(self.channels)
        self.bn_emb = nn.BatchNorm1d(args.emb_dims)
        self.conv_final = nn.Sequential(
            nn.Conv1d(total_ch, args.emb_dims, kernel_size=1, bias=False),
            self.bn_emb,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        points = x  # (B, C_in, N)

        # Compute adjacency once (full k neighbors)
        # Note: get_graph_feature returns flattened indices (B*N*k) when idx=None
        _, common_idx = get_graph_feature(points, k=self.k)

        features = []
        current_x = points  # (B, C, N)
        for i in range(self.num_layers):
            if self.use_sampling:
                # SamplingEdgeConv expects flattened common_idx so it can sample s neighbors
                x_out = self.layers[i](current_x, common_idx)
            else:
                feat_x, _ = get_graph_feature(current_x, k=self.k, idx=common_idx)
                x_out = self.layers[i](feat_x)
                x_out = self.bn_list[i](x_out)
                x_out = F.leaky_relu(x_out, negative_slope=0.2)
                x_out = x_out.max(dim=-1, keepdim=False)[0]

            current_x = x_out
            features.append(current_x)

        # Combine local features
        x = torch.cat(features, dim=1)
        x = self.conv_final(x)                      # (B, emb_dims, N)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)                  # (B, emb_dims*2)
        x = F.leaky_relu(self.bn4(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn5(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
