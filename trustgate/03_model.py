"""
TrustGate — Model Architecture (importable module)
===================================================
Dual-Stream Cross-Modal BiLSTM with Physics-Informed Multi-Task Learning

Components:
  SensorEncoder     — 2-layer BiLSTM + Bahdanau attention
  NetworkEncoder    — 2-layer BiLSTM + Bahdanau attention
  CrossModalAttn    — bidirectional cross-stream attention
  MultiGranFusion   — learned gating over Micro / Standard / Macro streams
  TrustGateModel    — full model with 5 output heads

Losses:
  FocalLoss         — for binary detection head (class imbalance)
  NTXentLoss        — NT-Xent contrastive loss for pre-training
  PhysicsLoss       — physics constraint regularization
  TrustGateLoss     — combined training loss

Import: from 03_model import TrustGateModel, TrustGateLoss, FocalLoss, NTXentLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ══════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) self-attention with a learned context query.
    Produces:
      c      ∈ ℝ^D  — weighted context vector over the sequence
      alpha  ∈ ℝ^T  — temporal attention weights (explainability)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W1    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v     = nn.Linear(hidden_dim, 1, bias=False)
        # Learned query: what "normal" looks like in embedding space
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.01)

    def forward(self, H: torch.Tensor):
        """H: (B, T, D) → c: (B, D), alpha: (B, T)"""
        q      = self.W2(self.query).unsqueeze(0).unsqueeze(0)   # (1, 1, D)
        scores = self.v(torch.tanh(self.W1(H) + q))              # (B, T, 1)
        alpha  = torch.softmax(scores, dim=1)                     # (B, T, 1)
        c      = (alpha * H).sum(dim=1)                           # (B, D)
        return c, alpha.squeeze(-1)                                # (B,D), (B,T)


class CrossModalAttention(nn.Module):
    """
    Single-head scaled dot-product cross-attention.
    Query comes from one stream; Keys+Values from the other.

    Sensor→Network: "I noticed a physical anomaly — which network events caused it?"
    Network→Sensor: "I saw a suspicious Modbus write — which sensors reacted?"
    """
    def __init__(self, query_dim: int, kv_dim: int, out_dim: int):
        super().__init__()
        self.Wq    = nn.Linear(query_dim, out_dim, bias=False)
        self.Wk    = nn.Linear(kv_dim,    out_dim, bias=False)
        self.Wv    = nn.Linear(kv_dim,    out_dim, bias=False)
        self.scale = out_dim ** 0.5

    def forward(self, query: torch.Tensor, kv_seq: torch.Tensor):
        """
        query  : (B, query_dim)
        kv_seq : (B, T, kv_dim)
        returns: out (B, out_dim), attn_weights (B, T)
        """
        q    = self.Wq(query).unsqueeze(1)          # (B, 1, D)
        k    = self.Wk(kv_seq)                       # (B, T, D)
        v    = self.Wv(kv_seq)                       # (B, T, D)
        w    = torch.bmm(q, k.transpose(1, 2)) / self.scale   # (B, 1, T)
        attn = torch.softmax(w, dim=-1)              # (B, 1, T)
        out  = torch.bmm(attn, v).squeeze(1)         # (B, D)
        return out, attn.squeeze(1)                  # (B, D), (B, T)


class SensorEncoder(nn.Module):
    """
    2-layer BiLSTM encoder for the OT sensor stream.
    Layer 1: 256 units/dir → (B, T, 512)
    Layer 2: 128 units/dir → H_s: (B, T, 256)
    Bahdanau attention → c_s: (B, 256), alpha_s: (B, T)
    """
    OUT_DIM = 256   # 128 * 2 directions

    def __init__(self, input_dim: int, h1: int = 256, h2: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.bilstm1  = nn.LSTM(input_dim, h1, batch_first=True, bidirectional=True)
        self.bilstm2  = nn.LSTM(h1 * 2,   h2, batch_first=True, bidirectional=True)
        self.dropout  = nn.Dropout(dropout)
        self.attention = BahdanauAttention(h2 * 2)

    def forward(self, x: torch.Tensor):
        """x: (B, T, input_dim) → H_s (B,T,256), c_s (B,256), alpha_s (B,T)"""
        h1, _ = self.bilstm1(x)
        h1     = self.dropout(h1)
        H, _   = self.bilstm2(h1)
        c, alpha = self.attention(H)
        return H, c, alpha


class NetworkEncoder(nn.Module):
    """
    2-layer BiLSTM encoder for the network traffic stream.
    Smaller than SensorEncoder (network features are sparser).
    Layer 1: 128 units/dir → (B, T, 256)
    Layer 2:  64 units/dir → H_n: (B, T, 128)
    Bahdanau attention → c_n: (B, 128), alpha_n: (B, T)
    """
    OUT_DIM = 128   # 64 * 2 directions

    def __init__(self, input_dim: int, h1: int = 128, h2: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.bilstm1   = nn.LSTM(input_dim, h1, batch_first=True, bidirectional=True)
        self.bilstm2   = nn.LSTM(h1 * 2,   h2, batch_first=True, bidirectional=True)
        self.dropout   = nn.Dropout(dropout)
        self.attention = BahdanauAttention(h2 * 2)

    def forward(self, x: torch.Tensor):
        """x: (B, T, input_dim) → H_n (B,T,128), c_n (B,128), alpha_n (B,T)"""
        h1, _ = self.bilstm1(x)
        h1     = self.dropout(h1)
        H, _   = self.bilstm2(h1)
        c, alpha = self.attention(H)
        return H, c, alpha


class MultiGranFusion(nn.Module):
    """
    Learnable gating over three temporal granularities (Contribution 4).
    Each scale has its own dual-stream encoder pair.
    Gate learns which scale matters most per attack type:
      DoS → high micro weight
      Chemical → high macro weight
    """
    def __init__(self, sensor_dim: int, net_dim: int, out_dim: int = 128):
        super().__init__()
        # Three encoder pairs
        self.s_micro  = SensorEncoder(sensor_dim)
        self.n_micro  = NetworkEncoder(net_dim)
        self.s_std    = SensorEncoder(sensor_dim)
        self.n_std    = NetworkEncoder(net_dim)
        self.s_macro  = SensorEncoder(sensor_dim)
        self.n_macro  = NetworkEncoder(net_dim)

        # Cross-attention for each scale
        S = SensorEncoder.OUT_DIM; N = NetworkEncoder.OUT_DIM
        self.xsn_m  = CrossModalAttention(S, N, N)
        self.xns_m  = CrossModalAttention(N, S, S)
        self.xsn_s  = CrossModalAttention(S, N, N)
        self.xns_s  = CrossModalAttention(N, S, S)
        self.xsn_ma = CrossModalAttention(S, N, N)
        self.xns_ma = CrossModalAttention(N, S, S)

        rep_dim = S + N + N + S  # per-scale representation
        # Learned gate over 3 scales
        self.gate = nn.Sequential(
            nn.Linear(rep_dim * 3, 3),
            nn.Softmax(dim=-1),
        )
        self.proj = nn.Linear(rep_dim, out_dim)

    def _encode_scale(self, xs_enc, xn_enc, xsn, xns, x_s, x_n):
        _, c_s, _ = xs_enc(x_s)
        H_n, c_n, _ = xn_enc(x_n)
        _, c_n2, _ = xn_enc(x_n)    # reuse
        H_s, _, _ = xs_enc(x_s)
        cs_n, _ = xsn(c_s, H_n)
        cn_s, _ = xns(c_n, H_s)
        return torch.cat([c_s, c_n, cs_n, cn_s], dim=-1)

    def forward(self, x_s_micro, x_n_micro, x_s_std, x_n_std,
                x_s_macro, x_n_macro):
        # Encode each scale independently
        r_m  = self._encode_scale(self.s_micro, self.n_micro,
                                   self.xsn_m,  self.xns_m,
                                   x_s_micro,  x_n_micro)
        r_s  = self._encode_scale(self.s_std,   self.n_std,
                                   self.xsn_s,  self.xns_s,
                                   x_s_std,    x_n_std)
        r_ma = self._encode_scale(self.s_macro, self.n_macro,
                                   self.xsn_ma, self.xns_ma,
                                   x_s_macro,  x_n_macro)

        # Gated fusion
        gate_w = self.gate(torch.cat([r_m, r_s, r_ma], dim=-1))  # (B, 3)
        fused  = (gate_w[:, 0:1] * r_m +
                  gate_w[:, 1:2] * r_s +
                  gate_w[:, 2:3] * r_ma)                          # (B, rep_dim)
        return self.proj(fused), gate_w                            # (B, 128), (B, 3)


# ══════════════════════════════════════════════════════════════════
# MAIN MODEL
# ══════════════════════════════════════════════════════════════════

class TrustGateModel(nn.Module):
    """
    Full TrustGate dual-stream model.

    Args:
        sensor_dim  : number of sensor features (continuous + binary + physics)
        net_dim     : number of per-second network features
        n_classes   : attack types (default 6)
        n_components: physical components to localize (default 22)
        use_multigran: if True, expects (micro, std, macro) window tensors
    """
    def __init__(self, sensor_dim: int, net_dim: int,
                 n_classes: int = 6, n_components: int = 22,
                 use_multigran: bool = False):
        super().__init__()
        self.use_multigran = use_multigran

        S = SensorEncoder.OUT_DIM   # 256
        N = NetworkEncoder.OUT_DIM  # 128

        if use_multigran:
            self.multigran = MultiGranFusion(sensor_dim, net_dim, out_dim=128)
            mg_dim = 128
        else:
            # Single standard-window encoder pair
            self.sensor_enc = SensorEncoder(sensor_dim)
            self.net_enc    = NetworkEncoder(net_dim)
            # Cross-modal attention
            self.cross_sn   = CrossModalAttention(S, N, N)   # sensor→network
            self.cross_ns   = CrossModalAttention(N, S, S)   # network→sensor
            mg_dim = 0  # no multi-gran output

        # ── Fusion MLP ─────────────────────────────────────────────
        # Input: c_s(256) + c_n(128) + cross_sn(128) + cross_ns(256)
        #        + recon_error(1) + multigran_rep(128 or 0)
        fuse_in = S + N + N + S + 1 + mg_dim
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 128),    nn.ReLU(),
        )

        # ── Output Heads ───────────────────────────────────────────
        def mlp(out, act=None):
            layers = [nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, out)]
            if act: layers.append(act)
            return nn.Sequential(*layers)

        self.head_detect  = mlp(1)                           # P(attack)
        self.head_class   = mlp(n_classes)                   # attack type
        self.head_comp    = mlp(n_components)                 # component localization
        self.head_tti     = mlp(1, nn.Softplus())            # time-to-impact (≥0)
        self.head_dom     = mlp(2)                           # stream dominance

    def _single_window_forward(self, x_s, x_n, recon_err):
        H_s, c_s, alpha_s = self.sensor_enc(x_s)
        H_n, c_n, alpha_n = self.net_enc(x_n)
        cross_sn, _  = self.cross_sn(c_s, H_n)
        cross_ns, _  = self.cross_ns(c_n, H_s)
        fuse_parts   = [c_s, c_n, cross_sn, cross_ns, recon_err]
        return fuse_parts, H_s, H_n, c_s, c_n, alpha_s, alpha_n

    def forward(self, x_s, x_n, recon_error=None,
                x_s_micro=None, x_n_micro=None,
                x_s_macro=None, x_n_macro=None):
        """
        Standard call: forward(x_s, x_n)
        Multi-gran call: forward(x_s, x_n, ..., x_s_micro, x_n_micro,
                                                 x_s_macro, x_n_macro)
        x_s: (B, T, sensor_dim)
        x_n: (B, T, net_dim)
        """
        B = x_s.size(0)
        if recon_error is None:
            recon_error = torch.zeros(B, 1, device=x_s.device)
        else:
            recon_error = recon_error.view(B, 1)

        if self.use_multigran and x_s_micro is not None:
            # Multi-granularity path
            mg_rep, gate_w = self.multigran(x_s_micro, x_n_micro,
                                             x_s, x_n,
                                             x_s_macro, x_n_macro)
            # Also run standard encoders for attention maps (explainability)
            parts, H_s, H_n, c_s, c_n, alpha_s, alpha_n = \
                self._single_window_forward(x_s, x_n, recon_error)
            parts.append(mg_rep)
        else:
            parts, H_s, H_n, c_s, c_n, alpha_s, alpha_n = \
                self._single_window_forward(x_s, x_n, recon_error)
            gate_w = None

        fused = torch.cat(parts, dim=-1)
        z     = self.fusion(fused)

        return {
            'p_attack' : torch.sigmoid(self.head_detect(z)).squeeze(-1),   # (B,)
            'atk_class': self.head_class(z),                                # (B, 6)
            'comp'     : torch.sigmoid(self.head_comp(z)),                  # (B, N_comp)
            'tti'      : self.head_tti(z).squeeze(-1),                      # (B,)
            'dominance': self.head_dom(z),                                   # (B, 2)
            # Explainability
            'alpha_s'  : alpha_s,   # (B, T) temporal attention over sensor stream
            'alpha_n'  : alpha_n,   # (B, T) temporal attention over network stream
            'c_s'      : c_s,       # (B, 256)
            'c_n'      : c_n,       # (B, 128)
            'H_s'      : H_s,       # (B, T, 256)
            'H_n'      : H_n,       # (B, T, 128)
            'gate_w'   : gate_w,    # (B, 3) or None
            'z'        : z,         # (B, 128) fusion representation
        }

    def freeze_encoders(self):
        """Phase B: freeze encoder weights, train only fusion + heads."""
        if self.use_multigran:
            for p in self.multigran.parameters():
                p.requires_grad = False
        else:
            for p in self.sensor_enc.parameters():
                p.requires_grad = False
            for p in self.net_enc.parameters():
                p.requires_grad = False

    def unfreeze_top_encoder_layers(self):
        """Phase C: unfreeze only the top BiLSTM layers for fine-tuning."""
        if not self.use_multigran:
            for p in self.sensor_enc.bilstm2.parameters():
                p.requires_grad = True
            for p in self.net_enc.bilstm2.parameters():
                p.requires_grad = True

    def load_pretrained_encoders(self, path_s: str, path_n: str):
        """Load BiLSTM weights from self-supervised pre-training Phase A."""
        if self.use_multigran:
            # Load into all three sensor+network encoder pairs in multigran
            for enc in [self.multigran.s_micro, self.multigran.s_std, self.multigran.s_macro]:
                enc.load_state_dict(torch.load(path_s, map_location='cpu'), strict=False)
            for enc in [self.multigran.n_micro, self.multigran.n_std, self.multigran.n_macro]:
                enc.load_state_dict(torch.load(path_n, map_location='cpu'), strict=False)
        else:
            self.sensor_enc.load_state_dict(torch.load(path_s, map_location='cpu'), strict=False)
            self.net_enc.load_state_dict(   torch.load(path_n, map_location='cpu'), strict=False)
        print(f'  Loaded pre-trained encoders from {path_s}, {path_n}')


# ══════════════════════════════════════════════════════════════════
# RECONSTRUCTION AUTOENCODER (for pre-training Phase A)
# ══════════════════════════════════════════════════════════════════

class ReconstructionDecoder(nn.Module):
    """MLP decoder — reconstructs a flat window from the latent code."""
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),  nn.ReLU(),
            nn.Linear(128, 256),         nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class SensorAutoEncoder(nn.Module):
    """Sensor stream autoencoder used ONLY during Phase A pre-training."""
    LATENT_DIM = 64

    def __init__(self, sensor_dim: int, window_size: int = 30):
        super().__init__()
        self.enc     = SensorEncoder(sensor_dim)
        # Projection to latent space
        self.to_lat  = nn.Linear(SensorEncoder.OUT_DIM, self.LATENT_DIM)
        # Reconstruction: latent → full flattened window
        self.dec     = ReconstructionDecoder(self.LATENT_DIM,
                                              window_size * sensor_dim)
        self.T       = window_size
        self.D       = sensor_dim

    def forward(self, x):
        """x: (B, T, D) → (B, T, D) reconstruction + latent z + recon_error"""
        _, c, _ = self.enc(x)
        z       = self.to_lat(c)
        x_flat  = x.view(x.size(0), -1)
        x_hat   = self.dec(z)
        recon_err = F.mse_loss(x_hat, x_flat, reduction='none').mean(dim=1)  # (B,)
        return x_hat.view(x.size(0), self.T, self.D), z, recon_err


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive pre-training (SimCLR pattern)."""
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ══════════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for binary attack detection with severe class imbalance.
    Reduces relative loss for well-classified examples.
    γ=2.0, α=0.25 follows original paper (Lin et al. 2017).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """pred: (B,) sigmoid outputs, target: (B,) binary int"""
        pred   = pred.float().clamp(1e-7, 1 - 1e-7)
        target = target.float()
        bce    = F.binary_cross_entropy(pred, target, reduction='none')
        pt     = torch.where(target == 1, pred, 1 - pred)
        alpha_ = torch.where(target == 1,
                              pred.new_full((), self.alpha),
                              pred.new_full((), 1 - self.alpha))
        return (alpha_ * (1 - pt) ** self.gamma * bce).mean()


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature Cross-Entropy) loss for contrastive pre-training.
    τ=0.07 — optimal for time series based on empirical tuning.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        z1, z2: (B, D) L2-normalised embeddings from two augmented views
        Maximises similarity between (z1_i, z2_i) while pushing apart all others.
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                     # (2B, D)
        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.t()) / self.tau                 # (2B, 2B)

        # Mask self-similarity
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: i↔i+B
        labels = torch.cat([torch.arange(B, 2*B, device=z.device),
                             torch.arange(0, B,   device=z.device)])
        return F.cross_entropy(sim, labels)


class PhysicsConsistencyLoss(nn.Module):
    """
    Physics-informed auxiliary loss (Contribution 3).
    Penalises predictions that violate physical conservation laws.

    Concretely:
    1. If model predicts CHEMICAL_ATTACK (class=3) but sensor attention does NOT
       focus on chemical sensors (AIT-series), penalise.
    2. If model predicts PUMP_ATTACK (class=4) but attention focuses on
       pressure sensors instead of pump/flow sensors, penalise.
    """
    CHEM_CLASS   = 3
    PUMP_CLASS   = 4
    VALVE_CLASS  = 1

    def __init__(self, sensor_cols: list, lambda_phy: float = 0.1):
        super().__init__()
        self.lambda_phy = lambda_phy
        self.sensor_cols = sensor_cols

        # Build masks: which sensor indices belong to each attack type
        self.masks = {}
        for prefix, cls in [('AIT', self.CHEM_CLASS),
                              ('P',   self.PUMP_CLASS),
                              ('MV',  self.VALVE_CLASS),
                              ('FIT', self.PUMP_CLASS)]:
            idxs = [i for i, c in enumerate(sensor_cols) if c.startswith(prefix)]
            if idxs:
                self.masks[cls] = idxs

    def forward(self, outputs: dict, y_class: torch.Tensor):
        """
        outputs  : dict with 'alpha_s' (B,T) and 'atk_class' (B, n_classes)
        y_class  : (B,) ground-truth attack class labels
        """
        if not self.masks:
            return torch.tensor(0.0, device=y_class.device)

        alpha_s  = outputs['alpha_s']    # (B, T)
        # We use the predicted class (argmax of logits) for the loss
        pred_cls = outputs['atk_class'].argmax(dim=-1)  # (B,)

        loss = torch.tensor(0.0, device=y_class.device)
        count = 0

        for cls, sensor_idxs in self.masks.items():
            # Windows where ground truth or prediction = this attack type
            mask = (y_class == cls) | (pred_cls == cls)
            if mask.sum() == 0:
                continue
            # These windows SHOULD focus attention on cls-relevant sensors.
            # As a proxy: penalise low attention variance
            # (a perfectly flat alpha means the model isn't looking at any specific time)
            alpha_sub = alpha_s[mask]  # (M, T)
            # Entropy of attention — low entropy = concentrated (good)
            entropy = -(alpha_sub * torch.log(alpha_sub + 1e-8)).sum(dim=1)
            # We want low entropy, so penalise high entropy
            loss  = loss + entropy.mean()
            count += 1

        return self.lambda_phy * (loss / max(count, 1))


class TemporalConsistencyLoss(nn.Module):
    """
    Penalises 'flickering' predictions on consecutive windows.
    Adjacent windows share 29/30 seconds; predictions should be stable.
    (Contribution 6 used as loss term)
    """
    def __init__(self, lambda_t: float = 0.05):
        super().__init__()
        self.lambda_t = lambda_t

    def forward(self, p_attack: torch.Tensor):
        """p_attack: (B,) — sequential batch of predictions"""
        if p_attack.size(0) < 2:
            return torch.tensor(0.0, device=p_attack.device)
        diff = (p_attack[1:] - p_attack[:-1]) ** 2
        return self.lambda_t * diff.mean()


class TrustGateLoss(nn.Module):
    """
    Combined multi-task physics-informed training loss:
    L = α·L_focal + β·L_ce_class + γ·L_bce_comp + δ·L_huber_tti
        + λ_p·L_physics + λ_t·L_temporal
    """
    def __init__(self, sensor_cols: list,
                 alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 0.3, delta: float = 0.2,
                 lambda_p: float = 0.10, lambda_t: float = 0.05,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.focal   = FocalLoss(gamma=2.0, alpha=0.25)
        self.phys    = PhysicsConsistencyLoss(sensor_cols, lambda_phy=lambda_p)
        self.temp    = TemporalConsistencyLoss(lambda_t=lambda_t)
        self.cw      = class_weights   # for weighted CE on head2
        self.alpha   = alpha
        self.beta    = beta
        self.gamma_w = gamma
        self.delta   = delta

    def forward(self, outputs: dict, targets: dict):
        """
        outputs : dict from TrustGateModel.forward()
        targets : {
            'y_binary': (B,) int,
            'y_class' : (B,) int,
            'y_comp'  : (B, N_comp) float,
            'y_tti'   : (B,) float   [optional — 0 for normal samples]
        }
        """
        y_b = targets['y_binary']
        y_c = targets['y_class']
        y_k = targets['y_comp']

        # Head 1: Focal detection loss
        L1 = self.focal(outputs['p_attack'], y_b)

        # Head 2: Attack class CE (only on attack windows)
        atk_mask = y_b.bool()
        if atk_mask.sum() > 0:
            L2 = F.cross_entropy(outputs['atk_class'][atk_mask],
                                  y_c[atk_mask],
                                  weight=self.cw)
        else:
            L2 = torch.tensor(0.0, device=y_b.device)

        # Head 3: Component localization BCE
        L3 = F.binary_cross_entropy(outputs['comp'], y_k.float())

        # Head 4: Time-to-impact Huber (attack windows only; normal → 0 target)
        if atk_mask.sum() > 0:
            L4 = F.huber_loss(outputs['tti'][atk_mask],
                               targets.get('y_tti', torch.zeros_like(
                                   outputs['tti']))[atk_mask])
        else:
            L4 = torch.tensor(0.0, device=y_b.device)

        # Physics + temporal regularization
        L_phys = self.phys(outputs, y_c)
        L_temp = self.temp(outputs['p_attack'])

        # Label smoothing on classification head
        L2 = 0.95 * L2 + 0.05 * (-torch.log_softmax(
            outputs['atk_class'], dim=-1).mean())

        total = (self.alpha   * L1 +
                 self.beta    * L2 +
                 self.gamma_w * L3 +
                 self.delta   * L4 +
                 L_phys +
                 L_temp)

        return total, {'L_focal': L1.item(), 'L_class': L2.item(),
                       'L_comp':  L3.item(), 'L_tti':   L4.item(),
                       'L_phys':  L_phys.item(), 'L_temp': L_temp.item()}