"""
capture.py
───────────
Captures live network traffic using tshark, computes flow-level
features matching CICIoT2023, and sends them to the running API.

Requirements:
    brew install wireshark      # installs tshark
    sudo chmod +x capture.py   # may need root for packet capture

Run:
    sudo uv run python capture.py --interface en0 --duration 30
"""
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from typing import Optional
import requests
import numpy as np
import shutil

TSHARK_PATH = shutil.which("tshark") or "/opt/homebrew/bin/tshark"

API_URL = "http://localhost:8000/predict"

# tshark fields that map to our 18 model features
# Format: -e field_name
TSHARK_FIELDS = [
    "frame.len",          # packet size
    "ip.ttl",             # Time_To_Live
    "ip.proto",           # Protocol Type (6=TCP, 17=UDP, 1=ICMP)
    "tcp.flags.fin",      # fin_flag_number
    "tcp.flags.syn",      # syn_flag_number
    "tcp.flags.reset",    # rst_flag_number
    "tcp.flags.push",     # psh_flag_number
    "tcp.flags.ack",      # ack_flag_number
    "frame.time_relative",# for IAT computation
    "ip.src",             # flow key
    "ip.dst",             # flow key
    "tcp.srcport",        # flow key
    "tcp.dstport",        # flow key
]


def build_tshark_cmd(interface: str, duration: int) -> list:
    fields = []
    for f in TSHARK_FIELDS:
        fields += ["-e", f]
    return [
        TSHARK_PATH,
        "-i", interface,
        "-a", f"duration:{duration}",
        "-T", "fields",
        "-E", "separator=,",
        "-E", "header=n",        # no header row
    ] + fields


class FlowAggregator:
    """
    Aggregates raw packets into flow-level features.
    A 'flow' is identified by (src_ip, dst_ip, src_port, dst_port, proto).
    Emits a flow when:
      - A FIN or RST flag is seen (flow ended cleanly)
      - The flow has accumulated 10+ packets (don't wait for clean close)
      - flush_all() is called at end of capture
    """

    def __init__(self, emit_after_packets: int = 10):
        self.flows = defaultdict(list)
        self.emit_after_packets = emit_after_packets

    def add_packet(self, fields: dict) -> Optional[dict]:
        """Add a packet to its flow. Returns flow features if flow is ready."""
        key = (
            fields.get("ip.src", ""),
            fields.get("ip.dst", ""),
            fields.get("tcp.srcport", "0"),
            fields.get("tcp.dstport", "0"),
            fields.get("ip.proto", "0"),
        )
        self.flows[key].append(fields)

        # Emit on FIN/RST (clean close)
        if fields.get("tcp.flags.fin") == "1" or fields.get("tcp.flags.reset") == "1":
            return self._compute_features(key)

        # Emit after enough packets accumulated — don't wait for clean close
        if len(self.flows[key]) >= self.emit_after_packets:
            return self._compute_features(key)

        return None

    def flush_all(self) -> list:
        """Compute features for all open flows (called at end of capture)."""
        results = []
        for key in list(self.flows.keys()):
            features = self._compute_features(key)
            if features:
                results.append(features)
        return results

    def _compute_features(self, key: tuple) -> Optional[dict]:
        packets = self.flows.pop(key, [])
        if len(packets) < 2:
            return None     # need at least 2 packets for a meaningful flow

        sizes   = [safe_float(p.get("frame.len", "0")) for p in packets]
        times   = [safe_float(p.get("frame.time_relative", "0")) for p in packets]
        proto   = safe_float(packets[0].get("ip.proto", "0"))

        duration = max(times) - min(times) if len(times) > 1 else 0.0
        n        = len(packets)

        def flag_sum(f): return sum(1 for p in packets if p.get(f, "").strip() == "1")

        ttl_vals = [safe_float(p.get("ip.ttl", "")) for p in packets if p.get("ip.ttl", "").strip()]
        avg_ttl  = float(np.mean(ttl_vals)) if ttl_vals else 0.0

        return {
            "Header_Length":    40.0,
            "Protocol Type":    proto,
            "Time_To_Live":     avg_ttl,
            "fin_flag_number":  flag_sum("tcp.flags.fin"),
            "syn_flag_number":  flag_sum("tcp.flags.syn"),
            "rst_flag_number":  flag_sum("tcp.flags.reset"),
            "psh_flag_number":  flag_sum("tcp.flags.push"),
            "ack_flag_number":  flag_sum("tcp.flags.ack"),
            "ack_count":        flag_sum("tcp.flags.ack"),
            "HTTP":             1.0 if packets[0].get("tcp.dstport", "").strip() == "80" else 0.0,
            "HTTPS":            1.0 if packets[0].get("tcp.dstport", "").strip() == "443" else 0.0,
            "TCP":              1.0 if proto == 6 else 0.0,
            "UDP":              1.0 if proto == 17 else 0.0,
            "ICMP":             1.0 if proto == 1 else 0.0,
            "Tot sum":          float(sum(sizes)),
            "Min":              float(min(sizes)),
            "AVG":              float(np.mean(sizes)),
            "Number":           float(n),
        }


def safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val.strip() else default
    except (ValueError, AttributeError):
        return default
    
def send_to_api(features: dict) -> dict:
    try:
        r = requests.post(API_URL, json=features, timeout=2)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        print("  ✗  API not reachable — is uvicorn running?")
        sys.exit(1)
    except Exception as e:
        return {"error": str(e)}


def print_result(features: dict, response: dict) -> None:
    result = response.get("result", {})
    pred   = result.get("prediction", "?")
    conf   = result.get("confidence", 0)
    attack = result.get("is_attack", False)
    low    = result.get("low_confidence", False)

    proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
    proto = proto_map.get(int(features.get("Protocol Type", 0)), "OTHER")
    n_pkts = int(features.get("Number", 0))

    flag = "🔴 ATTACK" if attack else "🟢 benign"
    warn = " ⚠ low confidence" if low else ""
    print(f"  [{proto:4s}] {n_pkts:4d} pkts | {pred:<15} {conf:.2f}  {flag}{warn}")


def run_capture(interface: str, duration: int, debug: bool = False) -> None:
    print(f"Capturing on {interface} for {duration}s — sending flows to {API_URL}")
    print("(Ctrl+C to stop early)\n")

    cmd = build_tshark_cmd(interface, duration)
    aggregator = FlowAggregator(emit_after_packets=5)
    processed = 0
    packet_count = 0

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            packet_count += 1
            if debug:
                print(f"  [pkt {packet_count}] {line[:80]}")

            values = line.split(",")
            if len(values) < len(TSHARK_FIELDS):
                if debug:
                    print(f"    ↳ skipped — only {len(values)} fields, need {len(TSHARK_FIELDS)}")
                continue

            fields = dict(zip(TSHARK_FIELDS, values))
            flow_features = aggregator.add_packet(fields)

            if flow_features:
                response = send_to_api(flow_features)
                print_result(flow_features, response)
                processed += 1

        proc.wait()
        if packet_count == 0:
            stderr = proc.stderr.read()
            print(f"⚠  tshark captured 0 packets.")
            print(f"   tshark stderr: {stderr[:300]}")
            print(f"   Try: tshark -D   to list available interfaces")

    except KeyboardInterrupt:
        print("\nStopped early — flushing open flows…")

    # Flush any flows still open at end of capture
    if packet_count > 0:
        print(f"\n{packet_count} packets captured. Flushing remaining flows…")
        for features in aggregator.flush_all():
            response = send_to_api(features)
            print_result(features, response)
            processed += 1

    print(f"\nDone. {processed} flows classified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live traffic → IDS API")
    parser.add_argument("--interface", default="en0",  help="Network interface (default: en0)")
    parser.add_argument("--duration",  default=30, type=int, help="Capture duration in seconds")
    parser.add_argument("--debug",     action="store_true",  help="Print raw tshark output")
    args = parser.parse_args()
    run_capture(args.interface, args.duration, args.debug)