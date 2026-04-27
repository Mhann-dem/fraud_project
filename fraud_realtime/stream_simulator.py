"""
=============================================================================
stream_simulator.py
=============================================================================
Simulates a live transaction feed for demo / load testing.
Reads from a PaySim CSV and replays rows as if they are arriving in real time.

Usage:
    python stream_simulator.py --data data/PS_*.csv --mode ws --rate 5
    python stream_simulator.py --data data/PS_*.csv --mode rest --rate 20
    python stream_simulator.py --demo --rate 3          # random synthetic txns

Modes:
  ws    — send each transaction over the WebSocket /ws/stream endpoint
  rest  — POST each transaction to /predict
  both  — simultaneous WS + REST

Options:
  --rate   Transactions per second (default: 5)
  --limit  Max transactions to send (default: unlimited)
  --demo   Generate random synthetic transactions without a CSV
  --host   API host (default: http://localhost:8000)
=============================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from datetime import datetime, timezone

import aiohttp
import pandas as pd
import numpy as np

# ── Column map from PaySim CSV header ────────────────────────────────────────
PAYSIM_COLS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]

TYPES = ["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"]


# =============================================================================
# Synthetic transaction generator (demo mode)
# =============================================================================

def synthetic_transaction(step: int) -> dict:
    """Generate one random mobile money transaction."""
    txn_type = random.choices(
        TYPES,
        weights=[0.22, 0.36, 0.08, 0.34, 0.01]
    )[0]
    amount       = round(random.lognormvariate(10, 1.5), 2)
    old_orig     = round(random.uniform(0, 500_000), 2)
    new_orig     = max(0.0, round(old_orig - amount, 2))
    old_dest     = round(random.uniform(0, 200_000), 2)
    new_dest     = round(old_dest + amount, 2)

    # Occasionally inject fraud-like patterns
    is_fraud_like = random.random() < 0.013
    if is_fraud_like:
        amount   = round(random.uniform(100_000, 5_000_000), 2)
        new_orig = 0.0

    return {
        "transaction_id":       str(uuid.uuid4()),
        "step":                 step,
        "type":                 txn_type,
        "amount":               amount,
        "oldbalanceOrg":        old_orig,
        "newbalanceOrig":       new_orig,
        "oldbalanceDest":       old_dest,
        "newbalanceDest":       new_dest,
        "transaction_velocity": random.randint(0, 10),
        "amount_deviation":     round(random.normalvariate(0, 2), 4),
        "balance_drop_flag":    int(new_orig == 0.0),
        "counterparty_spread":  random.randint(0, 8),
        "error_balance":        round(abs(old_orig + amount - new_orig), 2),
    }


def csv_transactions(csv_path: str):
    """Yield transaction dicts from a PaySim CSV, row by row."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    for _, row in df.iterrows():
        yield {
            "transaction_id":       str(uuid.uuid4()),
            "step":                 int(row.get("step", 0)),
            "type":                 str(row.get("type", "PAYMENT")).upper(),
            "amount":               float(row.get("amount", 0)),
            "oldbalanceOrg":        float(row.get("oldbalanceOrg", 0)),
            "newbalanceOrig":       float(row.get("newbalanceOrig", 0)),
            "oldbalanceDest":       float(row.get("oldbalanceDest", 0)),
            "newbalanceDest":       float(row.get("newbalanceDest", 0)),
            "transaction_velocity": 0.0,
            "amount_deviation":     0.0,
            "balance_drop_flag":    0,
            "counterparty_spread":  0.0,
            "error_balance":        abs(float(row.get("oldbalanceOrg", 0))
                                       + float(row.get("amount", 0))
                                       - float(row.get("newbalanceOrig", 0))),
        }


# =============================================================================
# REST sender
# =============================================================================

async def send_rest(session: aiohttp.ClientSession, host: str, txn: dict):
    url = f"{host}/predict"
    try:
        async with session.post(url, json=txn, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            data = await resp.json()
            band = data.get("risk_band", "?")
            prob = data.get("fraud_probability", 0)
            lat  = data.get("latency_ms", 0)
            flag = "🚨" if band == "HIGH" else ("⚠️ " if band == "MEDIUM" else "✅")
            print(f"  {flag} REST  | {band:<6} | p={prob:.4f} | {lat:.1f}ms | {txn['transaction_id'][:8]}")
    except Exception as e:
        print(f"  ❌ REST error: {e}")


# =============================================================================
# WebSocket sender
# =============================================================================

async def stream_websocket(host: str, txn_gen, rate: float, limit: int | None):
    ws_url = host.replace("http://", "ws://").replace("https://", "wss://") + "/ws/stream"
    print(f"\n🔌 Connecting WebSocket → {ws_url}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(ws_url, heartbeat=20) as ws:
                print("   Connected ✓\n")
                count = 0
                delay = 1.0 / rate

                async for txn in aiter(txn_gen):
                    if limit and count >= limit:
                        break

                    # Send transaction to the REST /predict endpoint
                    # (WebSocket is for *receiving* scored results from the server)
                    await send_rest(session, host, txn)

                    # Also read any pushed messages from the server
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=0.05)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("action") not in ("pong", "heartbeat"):
                                print(f"  📡 WS push: {data.get('risk_band')} "
                                      f"p={data.get('fraud_probability', 0):.4f}")
                    except asyncio.TimeoutError:
                        pass

                    count += 1
                    await asyncio.sleep(delay)

                print(f"\n  Sent {count} transactions.")
        except Exception as e:
            print(f"  ❌ WebSocket error: {e}")


async def aiter(gen):
    """Wrap a sync generator for use in async context."""
    for item in gen:
        yield item


# =============================================================================
# REST-only mode
# =============================================================================

async def stream_rest(host: str, txn_gen, rate: float, limit: int | None):
    delay = 1.0 / rate
    count = 0
    print(f"\n🚀 Sending to REST API → {host}/predict at {rate} tx/s\n")
    async with aiohttp.ClientSession() as session:
        async for txn in aiter(txn_gen):
            if limit and count >= limit:
                break
            await send_rest(session, host, txn)
            count += 1
            await asyncio.sleep(delay)
    print(f"\n  Sent {count} transactions.")


# =============================================================================
# Entry point
# =============================================================================

def make_generator(args):
    if args.demo:
        def _gen():
            step = 1
            while True:
                yield synthetic_transaction(step)
                step += 1
        return _gen()
    elif args.data:
        return csv_transactions(args.data)
    else:
        raise ValueError("Provide --data <csv_path> or --demo flag.")


async def main_async(args):
    gen = make_generator(args)

    if args.mode == "rest":
        await stream_rest(args.host, gen, args.rate, args.limit)
    elif args.mode == "ws":
        await stream_websocket(args.host, gen, args.rate, args.limit)
    elif args.mode == "both":
        gen2 = make_generator(args)
        await asyncio.gather(
            stream_rest(args.host, gen,  args.rate / 2, args.limit),
            stream_websocket(args.host, gen2, args.rate / 2, args.limit),
        )


def main():
    parser = argparse.ArgumentParser(description="Fraud detection stream simulator")
    parser.add_argument("--data",  type=str,   default=None,               help="PaySim CSV path")
    parser.add_argument("--demo",  action="store_true",                    help="Use synthetic random transactions")
    parser.add_argument("--mode",  type=str,   default="rest",             choices=["rest", "ws", "both"])
    parser.add_argument("--rate",  type=float, default=5.0,                help="Transactions per second")
    parser.add_argument("--limit", type=int,   default=None,               help="Max transactions to send")
    parser.add_argument("--host",  type=str,   default="http://localhost:8000")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════╗
║   Fraud Detection — Live Stream Simulator    ║
╠══════════════════════════════════════════════╣
║  Mode   : {args.mode:<35}║
║  Rate   : {str(args.rate) + ' tx/s':<35}║
║  Source : {'DEMO (synthetic)' if args.demo else (args.data or 'N/A'):<35}║
║  Target : {args.host:<35}║
╚══════════════════════════════════════════════╝
""")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
