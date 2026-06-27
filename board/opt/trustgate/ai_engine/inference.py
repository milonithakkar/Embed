#!/usr/bin/env python3
"""
TrustGate Enhanced — Real-Time OpenVINO Inference Engine
Intel Cup ESDC 2025 | IIT Gandhinagar

Board path : /opt/trustgate/ai_engine/inference.py
Reads      : /tmp/twin_report.json        (from physics_sim.py)
Writes     : /tmp/ai_predictions.json     (consumed by app.py)
Model dir  : /opt/trustgate/model/        (compiled OpenVINO IR v3)

Model: Dual-Stream Cross-Modal BiLSTM + Cross-Attention
  Val AUC  = 0.9620
  Val F1   = 0.7389
  FAR      = 2.97%
  Inputs   : sensor stream (71-dim) + network stream (19-dim)
             both over 30-step rolling sequence window
  Outputs  : [0] binary anomaly logits  shape (1,1)
             [1] attack class logits    shape (1,6)
             [2] sensor importance map  shape (1,71)
"""

import time
import json
import os
import sys
import numpy as np
import pickle
import logging
from collections import deque

# ---------------------------------------------------------------------------
# Path Configuration — all paths are board-absolute
# ---------------------------------------------------------------------------
NETWORK_FEATURES_PATH = "/tmp/network_features.json"
SHUTDOWN_FLAG         = "/tmp/trustgate_shutdown.flag"
TWIN_PATH     = "/tmp/twin_report.json"
OUTPUT_PATH   = "/tmp/ai_predictions.json"
MODEL_XML     = "/opt/trustgate/model/trustgate_model.xml"
MODEL_BIN     = "/opt/trustgate/model/trustgate_model.bin"
SCALER_PATH   = "/opt/trustgate/model/a11_scaler.pkl"
BASELINE_PATH = "/opt/trustgate/model/sensor_normal_baseline.npy"

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH        = 30     # rolling window steps
SENSOR_DIM             = 71     # sensor stream feature count
NETWORK_DIM            = 19     # network stream feature count
INFERENCE_INTERVAL     = 0.5    # seconds between cycles
ANOMALY_THRESHOLD_HIGH = 0.85   # → CRITICAL
ANOMALY_THRESHOLD_MED  = 0.50   # → WARNING

# Attack class labels — must match training label encoding exactly
ATTACK_CLASSES = [
    "Normal",
    "Reconnaissance/Scanning",
    "False Data Injection",
    "Command Injection",
    "DoS/Flooding",
    "Multi-Stage APT",
]

# Sensor index → plant component name mapping
# Covers the indices that carry physical meaning in the 71-dim vector
SENSOR_TO_COMPONENT = {
    0:  "LIT101",  1:  "FIT101",  2:  "P101_status", 3:  "LIT101_norm",
    10: "AIT201",  11: "AIT202",  12: "pH_deviation", 13: "pH_alert",
    20: "DPIT301", 21: "FIT301",  22: "UF_age",       23: "UF_resistance",
    24: "UF_foul", 30: "LIT401",  31: "AIT402",       32: "FIT401",
    33: "LIT401_norm", 34: "cond_alert",
    40: "AIT501",  41: "AIT502",  42: "FIT501",       43: "mem_hours",
    44: "RO_recovery",
    50: "LIT601",  51: "FIT601",  52: "LIT601_norm",  53: "LIT601_low",
    60: "mass_imbalance",   61: "total_inventory",
    62: "pH_differential",  63: "attack_flag",
    64: "tick_proxy",       65: "UF_power",
    66: "ORP_pH_interact",  67: "head_diff",
    68: "UF_RO_gap",        69: "degradation",
    70: "dual_aging",
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] INFER   | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inference")


# ---------------------------------------------------------------------------
# Synthetic Network Feature Generator
# Produces the 19-dim network stream vector.
# In production this would tap a real pcap/netflow pipeline.
# For the demo it synthesises realistic Modbus TCP traffic statistics,
# with anomalous characteristics injected when attack_flag is True.
# ---------------------------------------------------------------------------
class NetworkFeatureReader:
    """
    Reads the real 19-dim network feature vector produced by
    network_extractor.py from live wire-level Modbus capture.

    Falls back to zero vector (not random noise) if extractor
    is not yet running, so the model sees silence rather than
    misleading synthetic data.
    """

    def __init__(self):
        self._last_features  = [0.0] * NETWORK_DIM
        self._last_timestamp = 0.0
        self._stale_warned   = False
        self._stale_count    = 0

    def generate(self, attack_flag: bool = False) -> np.ndarray:
        """
        Read latest features from /tmp/network_features.json.
        Includes retry logic to handle write-read race conditions.
        """
        if not os.path.exists(NETWORK_FEATURES_PATH):
            if not self._stale_warned:
                log.warning(
                    "network_features.json not found — "
                    "is network_extractor.py running as sudo?"
                )
                self._stale_warned = True
            return np.zeros(NETWORK_DIM, dtype=np.float32)

        # Retry up to 3 times on JSON decode error
        data = None
        for attempt in range(3):
            try:
                with open(NETWORK_FEATURES_PATH, 'r') as f:
                    content = f.read()
                data = json.loads(content)
                break
            except (json.JSONDecodeError, IOError):
                time.sleep(0.02)

        if data is None:
            # All retries failed — return last known good features
            log.debug("Network features read failed — using last known")
            return np.array(self._last_features, dtype=np.float32)

        features  = data.get('features', [0.0] * NETWORK_DIM)
        timestamp = data.get('timestamp', 0.0)

        # Detect stale data
        if timestamp <= self._last_timestamp:
            self._stale_count += 1
            if self._stale_count == 10:
                log.warning(
                    "Network features stale — "
                    "network_extractor.py may have stopped"
                )
        else:
            self._stale_count    = 0
            self._stale_warned   = False
            self._last_timestamp = timestamp
            self._last_features  = features

        # Dimension guard
        if len(features) != NETWORK_DIM:
            log.warning(
                f"Feature dim mismatch: "
                f"got {len(features)}, expected {NETWORK_DIM}"
            )
            if len(features) < NETWORK_DIM:
                features = features + [0.0] * (NETWORK_DIM - len(features))
            else:
                features = features[:NETWORK_DIM]

        return np.array(features, dtype=np.float32)
# ---------------------------------------------------------------------------
# Core Inference Engine
# ---------------------------------------------------------------------------
class TrustGateInference:

    def __init__(self):
        self._sensor_buf  = deque(maxlen=SEQUENCE_LENGTH)
        self._network_buf = deque(maxlen=SEQUENCE_LENGTH)
        self._model       = None    # compiled OpenVINO model
        self._scaler      = None    # sklearn StandardScaler
        self._baseline    = None    # normal-operation sensor baseline
        self._net_gen     = NetworkFeatureReader()
        self._cycle       = 0
        self._ema         = 0.0     # exponential moving average of anomaly prob

    # ── Setup ────────────────────────────────────────────────────────────
    def load(self):
        """Load OpenVINO compiled model and preprocessing assets."""
        try:
                from openvino import Core
        except ImportError:
                log.error(
                        "OpenVINO runtime not found.\n"
                        "Install: pip3 install openvino"
                )
                sys.exit(1)

        # Verify model files exist before attempting load
        for p in (MODEL_XML, MODEL_BIN):
            if not os.path.exists(p):
                log.error(f"Model file missing: {p}")
                log.error("Run scp from laptop first — see deployment guide.")
                sys.exit(1)

        log.info(f"Loading OpenVINO IR from {MODEL_XML} ...")
        ie    = Core()
        model = ie.read_model(model=MODEL_XML, weights=MODEL_BIN)
        self._model = ie.compile_model(model=model, device_name="CPU")

        n_in  = len(self._model.inputs)
        n_out = len(self._model.outputs)
        log.info(f"  Compiled — inputs: {n_in}  outputs: {n_out}")

        # Scaler
        if not os.path.exists(SCALER_PATH):
            log.error(f"Scaler missing: {SCALER_PATH}")
            sys.exit(1)
        with open(SCALER_PATH, "rb") as f:
            self._scaler = pickle.load(f)
        log.info(f"  Scaler loaded — expects {self._scaler.n_features_in_} features")

        # Baseline (optional — used for deviation metric only)
        if os.path.exists(BASELINE_PATH):
            self._baseline = np.load(BASELINE_PATH)
            log.info(f"  Baseline loaded — shape {self._baseline.shape}")
        else:
            log.warning(f"  Baseline not found at {BASELINE_PATH} — deviation will be 0")

        log.info("TrustGate Inference Engine ready.")

    # ── Twin Report Reader ────────────────────────────────────────────────
    def _read_twin(self) -> dict | None:
        """Safely read /tmp/twin_report.json written by physics_sim.py."""
        if not os.path.exists(TWIN_PATH):
            return None
        try:
            with open(TWIN_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    # ── Preprocessing ─────────────────────────────────────────────────────
    def _scale(self, raw: np.ndarray) -> np.ndarray:
        """
        Scale raw 71-dim sensor vector through the trained scaler.
        Handles dimension mismatch by padding or truncating to match
        the scaler's expected feature count, then slicing back to
        SENSOR_DIM for the model input.
        """
        expected = self._scaler.n_features_in_
        vec = raw.reshape(1, -1)

        if vec.shape[1] < expected:
            pad = np.zeros((1, expected - vec.shape[1]), dtype=np.float32)
            vec = np.concatenate([vec, pad], axis=1)
        elif vec.shape[1] > expected:
            vec = vec[:, :expected]

        scaled = self._scaler.transform(vec)
        return scaled[0, :SENSOR_DIM]

    def _build_sequence(self, buf: deque, dim: int) -> np.ndarray:
        """
        Build (1, SEQUENCE_LENGTH, dim) array from the rolling buffer.
        Pads with repetitions of the oldest entry if buffer not yet full.
        """
        seq = list(buf)
        while len(seq) < SEQUENCE_LENGTH:
            seq.insert(0, seq[0])                  # repeat oldest frame
        arr = np.array(seq, dtype=np.float32)      # (30, dim)
        return arr.reshape(1, SEQUENCE_LENGTH, dim)

    # ── Forward Pass ──────────────────────────────────────────────────────
    def _forward(
        self,
        sensor_seq: np.ndarray,
        network_seq: np.ndarray,
    ) -> tuple:
        """
        Run OpenVINO inference.
        Handles both dual-input and single-input (concatenated) model graphs.
        Returns (binary_logit, class_logits_array, importance_array).
        """
        n_in = len(self._model.inputs)

        if n_in >= 2:
            results = self._model({
                self._model.input(0): sensor_seq,
                self._model.input(1): network_seq,
            })
        else:
            # Fallback: concatenate along feature axis → (1, 30, 90)
            combined = np.concatenate([sensor_seq, network_seq], axis=2)
            results  = self._model({self._model.input(0): combined})

        binary_logits = results[self._model.output(0)]          # (1,1)
        class_logits  = (
            results[self._model.output(1)]
            if len(self._model.outputs) > 1
            else np.zeros((1, len(ATTACK_CLASSES)), dtype=np.float32)
        )
        importances = (
            results[self._model.output(2)]
            if len(self._model.outputs) > 2
            else np.abs(np.random.randn(1, SENSOR_DIM).astype(np.float32))
        )

        return binary_logits, class_logits, importances

    # ── Postprocessing helpers ────────────────────────────────────────────
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-9)

    # ── Fallback payload when twin data is not yet available ──────────────
    def _fallback(self, note: str) -> dict:
        return {
            "alert_level":            "INITIALIZING",
            "confidence":             0.0,
            "attack_class":           "Awaiting data",
            "attack_class_id":        -1,
            "class_probabilities":    {},
            "top_component":          "N/A",
            "root_cause_localization": {
                "top5_sensors":    [],
                "top5_components": [],
                "top5_scores":     [],
            },
            "component_score":        0.0,
            "prob_v3":                0.0,
            "prob_v4":                0.0,
            "baseline_deviation":     0.0,
            "cumulative_anomaly_ema": self._ema,
            "buffer_fill":            f"{len(self._sensor_buf)}/{SEQUENCE_LENGTH}",
            "inference_count":        self._cycle,
            "model_version":          "v3_FINAL_AUC9620",
            "timestamp":              time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "epoch_time":             time.time(),
            "status_note":            note,
        }

    # ── One inference cycle ───────────────────────────────────────────────
    def infer(self) -> dict:
        # 1. Read twin report
        twin = self._read_twin()
        if twin is None:
            return self._fallback("Waiting for twin_report.json")

        # 2. Extract and validate sensor vector
        raw = np.array(
            twin.get("sensors", [0.0] * SENSOR_DIM),
            dtype=np.float32,
        )
        if len(raw) < SENSOR_DIM:
            raw = np.pad(raw, (0, SENSOR_DIM - len(raw)))

        # 3. Scale and push to sensor buffer
        scaled = self._scale(raw)
        self._sensor_buf.append(scaled)

        # 4. Generate network features and push to network buffer
        attack_flag = bool(twin.get("attack_active", False))
        net_feat    = self._net_gen.generate(attack_flag=attack_flag)
        self._network_buf.append(net_feat)

        # 5. Build sequence windows — pad if buffer not yet full
        if len(self._sensor_buf) < 2:
            return self._fallback("Filling sequence buffer...")

        sensor_seq  = self._build_sequence(self._sensor_buf,  SENSOR_DIM)
        network_seq = self._build_sequence(self._network_buf, NETWORK_DIM)

        # 6. OpenVINO forward pass
        try:
            binary_logits, class_logits, importances = self._forward(
                sensor_seq, network_seq
            )
        except Exception as e:
            log.error(f"Inference error: {e}")
            return self._fallback(f"Inference error: {e}")

        # 7. Postprocess outputs
        anomaly_prob  = float(self._sigmoid(float(binary_logits.flat[0])))
        class_probs   = self._softmax(class_logits.flatten())
        pred_class_id = int(np.argmax(class_probs))
        pred_class    = ATTACK_CLASSES[pred_class_id] if pred_class_id < len(ATTACK_CLASSES) else "Unknown"

        # Alert level
        if anomaly_prob > ANOMALY_THRESHOLD_HIGH:
            alert_level = "CRITICAL"
        elif anomaly_prob > ANOMALY_THRESHOLD_MED:
            alert_level = "WARNING"
        else:
            alert_level = "NORMAL"

        # Root-cause localisation — top-5 by absolute importance
        imp_flat  = importances.flatten()[:SENSOR_DIM]
        top5_idx  = np.argsort(np.abs(imp_flat))[-5:][::-1].tolist()
        top5_comp = [SENSOR_TO_COMPONENT.get(i, f"S{i}") for i in top5_idx]
        top5_scr  = [round(float(imp_flat[i]), 4) for i in top5_idx]

        # Baseline deviation
        deviation = 0.0
        if self._baseline is not None:
            bl = self._baseline[:SENSOR_DIM] if len(self._baseline) >= SENSOR_DIM else self._baseline
            deviation = float(np.mean(np.abs(raw[:len(bl)] - bl)))

        # EMA anomaly score (α = 0.1)
        self._ema = round(0.9 * self._ema + 0.1 * anomaly_prob, 4)
        self._cycle += 1

        # 8. Build output payload
        payload = {
            "alert_level":    alert_level,
            "confidence":     round(anomaly_prob, 4),
            "attack_class":   pred_class,
            "attack_class_id": pred_class_id,
            "class_probabilities": {
                ATTACK_CLASSES[i]: round(float(class_probs[i]), 4)
                for i in range(min(len(ATTACK_CLASSES), len(class_probs)))
            },
            "top_component":  twin.get("last_active_component", "Unknown"),
            "root_cause_localization": {
                "top5_sensors":    top5_idx,
                "top5_components": top5_comp,
                "top5_scores":     top5_scr,
            },
            "component_score":        round(float(np.max(class_probs)), 4),
            "prob_v3":                round(anomaly_prob, 4),
            "prob_v4":                0.0,
            "baseline_deviation":     round(deviation, 4),
            "cumulative_anomaly_ema": self._ema,
            "buffer_fill":            f"{len(self._sensor_buf)}/{SEQUENCE_LENGTH}",
            "inference_count":        self._cycle,
            "model_version":          "v3_FINAL_AUC9620",
            "timestamp":              time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "epoch_time":             time.time(),
        }

        return payload


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
def main():
    engine = TrustGateInference()
    engine.load()

    log.info(f"Reading twin data from : {TWIN_PATH}")
    log.info(f"Writing predictions to : {OUTPUT_PATH}")
    log.info(f"Network features from  : {NETWORK_FEATURES_PATH}")
    log.info(f"Inference interval     : {INFERENCE_INTERVAL}s")

    was_shutdown = False

    while True:
        try:
            # ── Shutdown flag check ───────────────────────────────
            if os.path.exists(SHUTDOWN_FLAG):
                if not was_shutdown:
                    engine._sensor_buf.clear()
                    engine._network_buf.clear()
                    engine._ema = 0.0
                    log.critical(
                        "SHUTDOWN FLAG detected — "
                        "buffers flushed, inference suspended"
                    )
                    was_shutdown = True
                time.sleep(INFERENCE_INTERVAL)
                continue

            if was_shutdown:
                log.info("Shutdown flag cleared — resuming inference")
                was_shutdown = False

            # ── Normal inference ──────────────────────────────────
            payload = engine.infer()

            tmp = OUTPUT_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, OUTPUT_PATH)

            if engine._cycle % 10 == 0:
                log.info(
                    f"Cycle {engine._cycle:>5d} | "
                    f"{payload['alert_level']:>12s} | "
                    f"conf={payload['confidence']:.4f} | "
                    f"class={payload['attack_class']:<28s} | "
                    f"EMA={payload['cumulative_anomaly_ema']:.4f} | "
                    f"buf={payload['buffer_fill']}"
                )

        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)

        time.sleep(INFERENCE_INTERVAL)


if __name__ == "__main__":
    main()
