# model.py
# Save as: C:\Users\HP\Downloads\trustgate\model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── HYPERPARAMETERS ───────────────────────────────────────────
SENSOR_INPUT   = 71
NETWORK_INPUT  = 19
WINDOW_SIZE    = 30
SENSOR_HIDDEN  = 128
NETWORK_HIDDEN = 64
DROPOUT        = 0.4
N_CLASSES      = 6
N_COMPONENTS   = 22

# ── LAYER NORM BiLSTM ─────────────────────────────────────────
class LayerNormBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out)
        out = self.drop(out)
        return out


# ── CROSS-MODAL ATTENTION ─────────────────────────────────────
class CrossModalAttention(nn.Module):
    def __init__(self, sensor_dim, network_dim, attn_dim=64):
        super().__init__()
        self.Wq_s2n = nn.Linear(sensor_dim,  attn_dim, bias=False)
        self.Wk_s2n = nn.Linear(network_dim, attn_dim, bias=False)
        self.Wv_s2n = nn.Linear(network_dim, attn_dim, bias=False)
        self.Wq_n2s = nn.Linear(network_dim, attn_dim, bias=False)
        self.Wk_n2s = nn.Linear(sensor_dim,  attn_dim, bias=False)
        self.Wv_n2s = nn.Linear(sensor_dim,  attn_dim, bias=False)
        self.scale  = attn_dim ** -0.5
        self.drop   = nn.Dropout(0.1)

    def forward(self, H_s, H_n):
        Q_s = self.Wq_s2n(H_s)
        K_n = self.Wk_s2n(H_n)
        V_n = self.Wv_s2n(H_n)
        scores_s2n = torch.bmm(
            Q_s, K_n.transpose(1, 2)) * self.scale
        attn_s2n   = F.softmax(scores_s2n, dim=-1)
        ctx_s2n    = torch.bmm(self.drop(attn_s2n), V_n)

        Q_n = self.Wq_n2s(H_n)
        K_s = self.Wk_n2s(H_s)
        V_s = self.Wv_n2s(H_s)
        scores_n2s = torch.bmm(
            Q_n, K_s.transpose(1, 2)) * self.scale
        attn_n2s   = F.softmax(scores_n2s, dim=-1)
        ctx_n2s    = torch.bmm(self.drop(attn_n2s), V_s)

        return (ctx_s2n, ctx_n2s,
                attn_s2n.mean(dim=1),
                attn_n2s.mean(dim=1))


# ── SENSOR IMPORTANCE ─────────────────────────────────────────
class SensorImportance(nn.Module):
    def __init__(self, fused_dim=128, n_sensors=71):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_sensors),
        )

    def forward(self, fused):
        logits = self.fc(fused)
        probs  = F.softmax(logits, dim=-1)
        return probs, logits


# ── MAIN MODEL ────────────────────────────────────────────────
class TrustGateModel(nn.Module):
    def __init__(self):
        super().__init__()
        sensor_out  = SENSOR_HIDDEN  * 2   # 256
        network_out = NETWORK_HIDDEN * 2   # 128

        self.sensor_enc  = LayerNormBiLSTM(
            SENSOR_INPUT,  SENSOR_HIDDEN,  2, DROPOUT)
        self.network_enc = LayerNormBiLSTM(
            NETWORK_INPUT, NETWORK_HIDDEN, 2, DROPOUT)
        self.cross_attn  = CrossModalAttention(
            sensor_out, network_out, attn_dim=64)

        fused_in = sensor_out + network_out + 64 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(DROPOUT * 0.75),
        )

        self.sensor_imp = SensorImportance(
            fused_dim=128, n_sensors=SENSOR_INPUT)

        self.head_binary = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1))
        self.head_class  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, N_CLASSES))
        self.head_comp   = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, N_COMPONENTS))

    def forward(self, x_sensor, x_network):
        H_s = self.sensor_enc(x_sensor)
        H_n = self.network_enc(x_network)

        ctx_s2n, ctx_n2s, attn_s2n, attn_n2s = \
            self.cross_attn(H_s, H_n)

        h_s_pool     = H_s.mean(dim=1)
        h_n_pool     = H_n.mean(dim=1)
        ctx_s2n_pool = ctx_s2n.mean(dim=1)
        ctx_n2s_pool = ctx_n2s.mean(dim=1)

        cat   = torch.cat([h_s_pool, h_n_pool,
                            ctx_s2n_pool, ctx_n2s_pool],
                           dim=-1)
        fused = self.fusion(cat)

        sensor_imp, sensor_imp_logits = \
            self.sensor_imp(fused)

        bin_logit   = self.head_binary(fused)
        cls_logits  = self.head_class(fused)
        comp_logits = self.head_comp(fused)

        return (bin_logit, cls_logits, comp_logits,
                attn_s2n, attn_n2s,
                sensor_imp, sensor_imp_logits)


# ── VERIFICATION ──────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("TrustGate Model — Architecture Verification")
    print("=" * 60)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model    = TrustGateModel().to(device)
    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,}")

    if not (1_000_000 <= n_params <= 1_100_000):
        print(f"WARNING: Expected ~1,035,748 parameters")
    else:
        print(f"Parameter count OK")

    B, T = 32, WINDOW_SIZE
    xs   = torch.randn(B, T, SENSOR_INPUT).to(device)
    xn   = torch.randn(B, T, NETWORK_INPUT).to(device)

    with torch.no_grad():
        out = model(xs, xn)

    names    = ['binary_logit', 'class_logits',
                'comp_logits', 'attn_s2n', 'attn_n2s',
                'sensor_imp', 'sensor_imp_logits']
    expected = [(B,1),(B,6),(B,22),(B,T),(B,T),(B,71),(B,71)]

    print(f"\nOutput shapes:")
    all_ok = True
    for name, tensor, exp in zip(names, out, expected):
        shape  = tuple(tensor.shape)
        ok     = (shape == exp)
        if not ok:
            all_ok = False
        print(f"  {name:22s}: {str(shape):15s} "
              f"(expected {str(exp)}) "
              f"[{'OK' if ok else 'FAIL'}]")

    s  = out[5].sum(dim=-1)
    ok = torch.allclose(s, torch.ones_like(s), atol=1e-4)
    print(f"\n  [{'OK' if ok else 'FAIL'}] "
          f"sensor_imp sums to 1.0 "
          f"(mean={s.mean():.4f})")

    print(f"\n{'='*60}")
    if all_ok:
        print("Model verified. Run: python train_v3.py")
    else:
        print("SHAPE ERRORS. Fix before training.")
    print("=" * 60)