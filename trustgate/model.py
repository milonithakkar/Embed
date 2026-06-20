# =================================================================
# model.py
# TrustGate Dual-Stream BiLSTM with Cross-Modal Attention
# Multi-task: binary detection + attack class + component localization
# Save as: C:\Users\HP\Downloads\trustgate\model.py
# =================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Architecture Constants ────────────────────────────────────────
SENSOR_FEATS    = 71      # from A12_windowed.npz
NETWORK_FEATS   = 19      # from smart extraction
WINDOW_SIZE     = 30
HIDDEN_SIZE     = 128     # per direction → 256 bidirectional
NUM_LAYERS      = 2
DROPOUT         = 0.3
FUSION_DIM      = 256

# Multi-task heads
NUM_ATTACK_CLASSES   = 6   # NORMAL + 5 attack types
NUM_COMPONENTS       = 22  # localization targets


# ── Single-Stream BiLSTM Encoder ─────────────────────────────────
class StreamEncoder(nn.Module):
    """
    BiLSTM encoder for one input stream.
    
    Used twice:
      - Sensor stream  (input_dim=71)
      - Network stream (input_dim=19)
    
    Output: (batch, 30, 256) — sequence of hidden states
    """
    def __init__(self, input_dim, name='stream'):
        super().__init__()
        self.name = name
        
        self.bilstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = HIDDEN_SIZE,
            num_layers    = NUM_LAYERS,
            batch_first   = True,
            bidirectional = True,
            dropout       = DROPOUT,
        )
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE * 2)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x):
        out, _ = self.bilstm(x)        # (B, 30, 256)
        out    = self.layer_norm(out)
        out    = self.dropout(out)
        return out


# ── Bidirectional Cross-Modal Attention ──────────────────────────
class CrossModalAttention(nn.Module):
    """
    BIDIRECTIONAL cross-attention between sensor and network streams.
    
    Computes BOTH directions:
      sensor → network : "Which network events explain this sensor anomaly?"
      network → sensor : "Which sensor changes did this network event cause?"
    
    This is the architectural novelty for the paper.
    No prior SWaT paper does bidirectional fusion at this granularity.
    
    Outputs:
      ctx_s   : (B, 256)  — sensor-side cross-attended context
      ctx_n   : (B, 256)  — network-side cross-attended context  
      attn_sn : (B, 30)   — attention weights sensor→network
      attn_ns : (B, 30)   — attention weights network→sensor
    """
    def __init__(self, hidden_dim=FUSION_DIM):
        super().__init__()
        
        # Sensor → Network direction
        self.W_q_s2n = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k_s2n = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_s2n   = nn.Linear(hidden_dim, 1,          bias=False)
        
        # Network → Sensor direction
        self.W_q_n2s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k_n2s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_n2s   = nn.Linear(hidden_dim, 1,          bias=False)
        
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, H_s, H_n):
        """
        H_s : (B, 30, 256)  — sensor hidden states sequence
        H_n : (B, 30, 256)  — network hidden states sequence
        """
        # ── Direction 1: Sensor attends to Network ────────────────
        # Use sensor's last timestep as query
        q_s = H_s[:, -1, :].unsqueeze(1)        # (B, 1, 256)
        
        # Attend to network sequence
        q_proj = self.W_q_s2n(q_s)              # (B, 1, 256)
        k_proj = self.W_k_s2n(H_n)              # (B, 30, 256)
        
        energy_s2n   = torch.tanh(q_proj + k_proj)  # (B, 30, 256)
        energy_s2n   = self.dropout(energy_s2n)
        scores_s2n   = self.v_s2n(energy_s2n).squeeze(-1)  # (B, 30)
        attn_s2n     = F.softmax(scores_s2n, dim=-1)       # (B, 30)
        
        ctx_s = (attn_s2n.unsqueeze(-1) * H_n).sum(dim=1)  # (B, 256)
        
        # ── Direction 2: Network attends to Sensor ────────────────
        q_n = H_n[:, -1, :].unsqueeze(1)        # (B, 1, 256)
        
        q_proj = self.W_q_n2s(q_n)              # (B, 1, 256)
        k_proj = self.W_k_n2s(H_s)              # (B, 30, 256)
        
        energy_n2s = torch.tanh(q_proj + k_proj)
        energy_n2s = self.dropout(energy_n2s)
        scores_n2s = self.v_n2s(energy_n2s).squeeze(-1)
        attn_n2s   = F.softmax(scores_n2s, dim=-1)
        
        ctx_n = (attn_n2s.unsqueeze(-1) * H_s).sum(dim=1)  # (B, 256)
        
        return ctx_s, ctx_n, attn_s2n, attn_n2s


# ── Sensor Attention (per-sensor importance) ─────────────────────
class SensorAttribution(nn.Module):
    """
    Computes per-sensor importance for explainability.
    Uses temporal attention from cross-modal layer to weight pooling.
    
    Output: (B, 71) — softmax over sensors
    """
    def __init__(self, num_sensors=SENSOR_FEATS):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, num_sensors),
        )

    def forward(self, H_s, attn_weights):
        """
        H_s          : (B, 30, 256)
        attn_weights : (B, 30)  — from cross-attention
        """
        w      = attn_weights.unsqueeze(-1)         # (B, 30, 1)
        pooled = (w * H_s).sum(dim=1)               # (B, 256)
        scores = self.proj(pooled)                   # (B, 71)
        return F.softmax(scores, dim=-1)


# ── Multi-Task Heads ─────────────────────────────────────────────
class MultiTaskHead(nn.Module):
    """
    3 prediction heads from fused representation:
    
    Head 1: Binary detection      → P(attack)
    Head 2: Attack classification → P(class | attack)
    Head 3: Component localization → P(each component involved)
    """
    def __init__(self, input_dim):
        super().__init__()
        
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        
        # Head 1: Binary attack/normal
        self.head_binary = nn.Linear(128, 1)
        
        # Head 2: 6-way attack class
        self.head_class = nn.Linear(128, NUM_ATTACK_CLASSES)
        
        # Head 3: Multi-label component
        self.head_components = nn.Linear(128, NUM_COMPONENTS)

    def forward(self, fused):
        features = self.shared(fused)                # (B, 128)
        
        binary_logit  = self.head_binary(features)        # (B, 1)
        class_logits  = self.head_class(features)         # (B, 6)
        comp_logits   = self.head_components(features)    # (B, 22)
        
        return binary_logit, class_logits, comp_logits


# ── Full TrustGate Model ─────────────────────────────────────────
class TrustGateModel(nn.Module):
    """
    Complete dual-stream BiLSTM with bidirectional cross-attention
    and multi-task heads.
    
    Forward returns 6 tensors:
      binary_logit : (B, 1)   — for BCE loss
      class_logits : (B, 6)   — for CE loss
      comp_logits  : (B, 22)  — for BCE multi-label loss
      attn_s2n     : (B, 30)  — sensor→network attention (paper viz)
      attn_n2s     : (B, 30)  — network→sensor attention (paper viz)
      sensor_imp   : (B, 71)  — per-sensor importance (dashboard)
    """
    def __init__(self):
        super().__init__()
        
        # Encoders
        self.sensor_enc  = StreamEncoder(SENSOR_FEATS,  'sensor')
        self.network_enc = StreamEncoder(NETWORK_FEATS, 'network')
        
        # Cross-modal attention (bidirectional)
        self.cross_attn = CrossModalAttention()
        
        # Sensor importance attribution
        self.sensor_imp = SensorAttribution()
        
        # Fusion + multi-task heads
        # Input: [sensor_last, network_last, ctx_s, ctx_n] = 4 × 256 = 1024
        self.heads = MultiTaskHead(input_dim=HIDDEN_SIZE * 2 * 4)
        
        self._init_weights()

    def _init_weights(self):
        """Xavier for linear, orthogonal for LSTM recurrent weights."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for pname, param in module.named_parameters():
                    if 'weight_ih' in pname:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in pname:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in pname:
                        nn.init.zeros_(param.data)
                        # Set forget gate bias to 1 (helps long sequences)
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)

    def forward(self, x_sensor, x_network):
        """
        x_sensor  : (B, 30, 71)
        x_network : (B, 30, 19)
        """
        # Encode both streams
        H_s = self.sensor_enc(x_sensor)         # (B, 30, 256)
        H_n = self.network_enc(x_network)        # (B, 30, 256)
        
        # Bidirectional cross-attention
        ctx_s, ctx_n, attn_s2n, attn_n2s = self.cross_attn(H_s, H_n)
        # ctx_s, ctx_n : (B, 256)
        # attn_s2n, attn_n2s : (B, 30)
        
        # Per-sensor importance (uses sensor→network attention)
        sensor_imp = self.sensor_imp(H_s, attn_s2n)  # (B, 71)
        
        # Build fused representation
        sensor_last  = H_s[:, -1, :]              # (B, 256)
        network_last = H_n[:, -1, :]              # (B, 256)
        
        fused = torch.cat(
            [sensor_last, network_last, ctx_s, ctx_n],
            dim=-1
        )                                          # (B, 1024)
        
        # Multi-task predictions
        binary_logit, class_logits, comp_logits = self.heads(fused)
        
        return (
            binary_logit,    # (B, 1)
            class_logits,    # (B, 6)
            comp_logits,     # (B, 22)
            attn_s2n,        # (B, 30)
            attn_n2s,        # (B, 30)
            sensor_imp,      # (B, 71)
        )


# ── Verification ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*60)
    print("TrustGate Model — Architecture Verification")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = TrustGateModel().to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Estimated size  : {total_params * 4 / (1024**2):.1f} MB")
    
    # Per-module breakdown
    print(f"\nPer-module parameters:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:15s}: {params:>10,}")
    
    # Forward pass test
    print(f"\nRunning forward pass (batch=32)...")
    x_s = torch.randn(32, 30, SENSOR_FEATS).to(device)
    x_n = torch.randn(32, 30, NETWORK_FEATS).to(device)
    
    with torch.no_grad():
        out = model(x_s, x_n)
    
    print(f"\nOutput shapes:")
    print(f"  binary_logit : {tuple(out[0].shape)}  (expected (32, 1))")
    print(f"  class_logits : {tuple(out[1].shape)}  (expected (32, 6))")
    print(f"  comp_logits  : {tuple(out[2].shape)}  (expected (32, 22))")
    print(f"  attn_s2n     : {tuple(out[3].shape)}  (expected (32, 30))")
    print(f"  attn_n2s     : {tuple(out[4].shape)}  (expected (32, 30))")
    print(f"  sensor_imp   : {tuple(out[5].shape)}  (expected (32, 71))")
    
    # Shape assertions
    assert out[0].shape == (32, 1)
    assert out[1].shape == (32, 6)
    assert out[2].shape == (32, 22)
    assert out[3].shape == (32, 30)
    assert out[4].shape == (32, 30)
    assert out[5].shape == (32, 71)
    print(f"\n  [OK] All shapes correct")
    
    # Attention sums to 1
    assert torch.allclose(out[3].sum(-1), torch.ones(32).to(device), atol=1e-4)
    assert torch.allclose(out[4].sum(-1), torch.ones(32).to(device), atol=1e-4)
    assert torch.allclose(out[5].sum(-1), torch.ones(32).to(device), atol=1e-4)
    print(f"  [OK] All attention/importance scores sum to 1.0")
    
    if device.type == 'cuda':
        mem_mb = torch.cuda.memory_allocated(device) / (1024**2)
        print(f"\nGPU memory: {mem_mb:.1f} MB")
    
    print("\n" + "="*60)
    print("Model verified. Run: python train.py")
    print("="*60)