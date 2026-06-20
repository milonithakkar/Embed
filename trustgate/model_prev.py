"""
TrustGate — Dual-Stream BiLSTM with Bahdanau Attention
Stream 1: Sensor  (B, 30, 44)  → BiLSTM → Attention → context (B, 256)
Stream 2: Network (B, 30, 132) → BiLSTM → Attention → context (B, 256)
Fusion: Concat → MLP → Binary head + Stage head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ATTACK_CLASSES = {
    0: 'NORMAL',
    1: 'CHEMICAL',
    2: 'PRESSURE',
    3: 'FLOW_TAMPER',
    4: 'PUMP_DOS',
    5: 'VALVE_ATTACK'
}


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v       = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(self, hidden_states, query):
        # hidden_states: (B, T, H) | query: (B, H)
        scores  = self.v(torch.tanh(
            self.W_query(query.unsqueeze(1)) + self.W_key(hidden_states)
        )).squeeze(-1)                                          # (B, T)
        weights = F.softmax(scores, dim=-1)                    # (B, T)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (B, H)
        return context, weights


class StreamEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = BahdanauAttention(hidden_dim * 2)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        out, (hn, _) = self.bilstm(x)                         # (B, T, H*2)
        query        = torch.cat([hn[-2], hn[-1]], dim=-1)     # (B, H*2)
        context, weights = self.attention(out, query)
        return self.dropout(context), weights


class TrustGateModel(nn.Module):
    def __init__(self,
                 sensor_dim=44, network_dim=132,
                 hidden_dim=128, num_layers=2,
                 dropout=0.3,   num_stages=6):
        super().__init__()
        self.sensor_enc  = StreamEncoder(sensor_dim,  hidden_dim, num_layers, dropout)
        self.network_enc = StreamEncoder(network_dim, hidden_dim, num_layers, dropout)

        fusion_in = hidden_dim * 4   # 512

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(128, 1)         # is_attack
        self.stage_head  = nn.Linear(128, num_stages) # attack type

    def forward(self, x_sensor, x_network):
        s_ctx, s_w = self.sensor_enc(x_sensor)
        n_ctx, n_w = self.network_enc(x_network)
        fused      = self.fusion(torch.cat([s_ctx, n_ctx], dim=-1))
        return self.binary_head(fused), self.stage_head(fused), s_w, n_w

    def predict(self, x_sensor, x_network, threshold=0.5):
        self.eval()
        with torch.no_grad():
            b_logit, s_logits, s_w, n_w = self.forward(x_sensor, x_network)
            confidence = torch.sigmoid(b_logit).squeeze(-1)
            is_attack  = (confidence >= threshold).long()
            stage      = torch.argmax(torch.softmax(s_logits, dim=-1), dim=-1)
            dominant   = torch.where(
                s_w.mean(-1) > n_w.mean(-1),
                torch.zeros_like(is_attack),   # 0 = SENSOR
                torch.ones_like(is_attack)     # 1 = NETWORK
            )
        return {
            'is_attack':  is_attack,
            'confidence': confidence,
            'stage':      stage,
            'dominant':   dominant,
            's_weights':  s_w,
            'n_weights':  n_w,
        }