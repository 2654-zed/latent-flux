"""Fetch prediction market prices from Polymarket and Kalshi.

Downloads hourly price snapshots for matched market pairs across both
platforms.  Saves to pma/data/pma_prices.json and pma/data/market_pairs.json.

Both APIs are public — no API key required.

Polymarket:
    Gamma API:  GET https://gamma-api.polymarket.com/events
    CLOB hist:  GET https://clob.polymarket.com/prices-history?market={clobTokenId}&interval=max&fidelity=60

Kalshi:
    Events:     GET https://api.elections.kalshi.com/trade-api/v2/events
    Markets:    GET https://api.elections.kalshi.com/trade-api/v2/markets?event_ticker={t}
    Trades:     GET https://api.elections.kalshi.com/trade-api/v2/markets/trades?ticker={t}
    Snapshot:   GET https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}

Usage:
    python scripts/fetch_pma_prices.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "pma" / "data"
PRICES_PATH = OUTPUT_DIR / "pma_prices.json"
PAIRS_PATH = OUTPUT_DIR / "market_pairs.json"

_SPORTS_CATS = {"sports"}

_STOPWORDS = {
    "will", "the", "a", "an", "in", "by", "of", "to", "be", "is",
    "are", "at", "or", "and", "for", "on", "its", "this", "that",
    "yes", "no", "above", "below", "over", "under", "before",
    "after", "than", "next", "new", "first", "any", "when", "who",
    "what", "how", "become", "get", "has", "have", "been", "not",
    "win", "released", "end", "round", "june", "march", "december",
    "january", "2026", "2025", "2027", "2028", "2029", "2030",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-z][a-z0-9]{2,}\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _get_json(url: str, retries: int = 2) -> dict | list | None:
    for attempt in range(retries + 1):
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "latent-flux-pma/1.0")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except (urllib.error.URLError, json.JSONDecodeError):
            return None
    return None


# ── Polymarket (Gamma API) ────────────────────────────────────────

def fetch_polymarket_events() -> list[dict]:
    """Return flat list of Polymarket markets from the Gamma events API."""
    print("Fetching Polymarket events (Gamma API)...", end=" ", flush=True)
    data = _get_json(
        "https://gamma-api.polymarket.com/events?closed=false&active=true&limit=200"
    )
    if not isinstance(data, list) or not data:
        print("FAILED")
        return []
    print(f"{len(data)} events")

    markets: list[dict] = []
    for ev in data:
        ev_title = ev.get("title", "")
        for m in ev.get("markets", []):
            q = m.get("question") or m.get("title") or ""
            clob_raw = m.get("clobTokenIds", "")
            try:
                clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
            except json.JSONDecodeError:
                clob_ids = []
            if not clob_ids or not q:
                continue
            markets.append({
                "_title": q,
                "_event_title": ev_title,
                "_id": clob_ids[0],       # first CLOB token = YES outcome
                "_condition_id": m.get("conditionId", ""),
                "_volume": float(m.get("volumeNum", 0) or 0),
            })
    print(f"  Total Polymarket markets with CLOB IDs: {len(markets)}")
    return markets


# ── Kalshi (Events → Markets) ─────────────────────────────────────

def fetch_kalshi_events() -> list[dict]:
    """Return flat list of non-sports Kalshi markets via events API."""
    print("Fetching Kalshi events...", end=" ", flush=True)
    data = _get_json(
        "https://api.elections.kalshi.com/trade-api/v2/events?limit=200&status=open"
    )
    if not isinstance(data, dict):
        print("FAILED")
        return []
    events = data.get("events", [])
    non_sports = [e for e in events
                  if e.get("category", "").lower() not in _SPORTS_CATS]
    print(f"{len(events)} events, {len(non_sports)} non-sports")

    markets: list[dict] = []
    for i, ev in enumerate(non_sports):
        eticker = ev.get("event_ticker", "")
        if not eticker:
            continue
        mdata = _get_json(
            f"https://api.elections.kalshi.com/trade-api/v2/markets"
            f"?limit=50&event_ticker={eticker}"
        )
        if isinstance(mdata, dict):
            for m in mdata.get("markets", []):
                title = m.get("title") or m.get("short_title") or ""
                ticker = m.get("ticker", "")
                if title and ticker:
                    markets.append({
                        "_title": title,
                        "_event_title": ev.get("title", ""),
                        "_id": ticker,
                        "_category": ev.get("category", ""),
                        "_last_price": m.get("last_price"),
                        "_yes_bid": m.get("yes_bid"),
                        "_yes_ask": m.get("yes_ask"),
                        "_volume": m.get("volume", 0),
                    })
        if (i + 1) % 5 == 0:
            time.sleep(1.5)
            if (i + 1) % 10 == 0:
                print(f"  ... {i+1}/{len(non_sports)} events "
                      f"({len(markets)} markets)", flush=True)
        else:
            time.sleep(0.4)

    print(f"  Total Kalshi non-sports markets: {len(markets)}")
    return markets


# ── Matching ──────────────────────────────────────────────────────

_GENERIC = {
    "united", "states", "president", "presidential", "election",
    "million", "billion", "year", "country", "world", "market",
    "price", "party", "government", "national",
}


def match_markets(
    poly: list[dict], kalshi: list[dict], min_overlap: int = 2
) -> list[dict]:
    """Greedy best-overlap matching, excluding generic-only overlaps."""
    pairs: list[dict] = []
    used_k: set[str] = set()
    used_p: set[str] = set()

    # Build kalshi token index for speed
    k_tokens = []
    for km in kalshi:
        combined = f"{km['_title']} {km.get('_event_title', '')}"
        k_tokens.append(_tokenize(combined))

    for pm in poly:
        pid = pm["_id"]
        if pid in used_p:
            continue
        p_text = f"{pm['_title']} {pm.get('_event_title', '')}"
        p_toks = _tokenize(p_text)

        best_i, best_n = -1, 0
        for j, km in enumerate(kalshi):
            kid = km["_id"]
            if kid in used_k:
                continue
            overlap = p_toks & k_tokens[j]
            real = overlap - _GENERIC
            n = len(real)
            if n >= min_overlap and n > best_n:
                best_n = n
                best_i = j

        if best_i >= 0:
            km = kalshi[best_i]
            slug = re.sub(r"[^a-z0-9_]", "_",
                          re.sub(r"[^a-z0-9 ]", "", pm["_title"].lower())[:40])
            pairs.append({
                "market_id": slug,
                "polymarket_id": pid,
                "kalshi_ticker": km["_id"],
                "description": pm["_title"],
                "_kalshi_title": km["_title"],
                "_overlap": best_n,
                "_poly_volume": pm.get("_volume", 0),
                "_kalshi_last": km.get("_last_price"),
            })
            used_k.add(km["_id"])
            used_p.add(pid)

    pairs.sort(key=lambda p: p["_overlap"], reverse=True)
    return pairs


# ── Price history ─────────────────────────────────────────────────

def fetch_poly_history(clob_token_id: str) -> list[dict]:
    """Hourly Polymarket prices via CLOB prices-history endpoint."""
    url = (
        f"https://clob.polymarket.com/prices-history"
        f"?market={clob_token_id}&interval=max&fidelity=60"
    )
    data = _get_json(url)
    if not isinstance(data, dict):
        return []
    return data.get("history", [])


def fetch_kalshi_trades(ticker: str, max_pages: int = 5) -> list[dict]:
    """Fetch recent Kalshi trades, paginated."""
    trades: list[dict] = []
    cursor = ""
    for _ in range(max_pages):
        url = (
            f"https://api.elections.kalshi.com/trade-api/v2/markets/trades"
            f"?ticker={ticker}&limit=100"
        )
        if cursor:
            url += f"&cursor={cursor}"
        data = _get_json(url)
        if not isinstance(data, dict):
            break
        batch = data.get("trades", [])
        if not batch:
            break
        trades.extend(batch)
        cursor = data.get("cursor", "")
        if not cursor:
            break
        time.sleep(0.4)
    return trades


def kalshi_trades_to_hourly(trades: list[dict]) -> list[dict]:
    """Aggregate Kalshi trades into hourly VWAP candles."""
    if not trades:
        return []

    buckets: dict[int, list[tuple[float, int]]] = {}
    for t in trades:
        ts_str = t.get("created_time", "")
        yes_price = t.get("yes_price", 0)
        count = t.get("count", 1)
        if not ts_str or not yes_price:
            continue
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            epoch = int(dt.timestamp())
        except (ValueError, OSError):
            continue
        hour = epoch - (epoch % 3600)
        price = yes_price / 100.0 if yes_price > 1 else yes_price
        buckets.setdefault(hour, []).append((price, count))

    candles = []
    for hour in sorted(buckets):
        entries = buckets[hour]
        total_vol = sum(c for _, c in entries)
        if total_vol == 0:
            continue
        vwap = sum(p * c for p, c in entries) / total_vol
        candles.append({"t": hour, "p": round(vwap, 4), "vol": total_vol})
    return candles


# ── Main ──────────────────────────────────────────────────────────

def main() -> int:
    if "--help" in sys.argv:
        print(__doc__)
        return 0

    # 1. Fetch market lists
    poly = fetch_polymarket_events()
    if not poly:
        print("Could not fetch Polymarket markets.")
        return 1
    time.sleep(0.5)

    kalshi = fetch_kalshi_events()
    if not kalshi:
        print("Could not fetch Kalshi markets.")
        return 1
    time.sleep(0.5)

    # 2. Match
    pairs = match_markets(poly, kalshi)
    print(f"\nMatched {len(pairs)} market pairs")

    if not pairs:
        print("No matching markets found.")
        return 1

    # Show top matches
    print("\nTop matched pairs:")
    for p in pairs[:15]:
        print(f"  [{p['_overlap']}] {p['description'][:60]}")
        print(f"      <-> {p['_kalshi_title'][:60]}")

    # 3. Fetch price history (limit to top 20 by overlap quality)
    top_pairs = pairs[:20]
    all_records: list[dict] = []
    successful = 0

    for pair in top_pairs:
        desc = pair["description"][:50]
        print(f"\n-- {desc}")

        # Polymarket history
        print(f"  Poly...", end=" ", flush=True)
        poly_hist = fetch_poly_history(pair["polymarket_id"])
        poly_count = 0
        if poly_hist:
            for h in poly_hist:
                ts = int(h.get("t", 0))
                price = float(h.get("p", 0))
                if ts > 0 and 0 <= price <= 1:
                    all_records.append({
                        "timestamp": ts,
                        "market_id": pair["market_id"],
                        "platform": "polymarket",
                        "yes_price": round(price, 4),
                        "volume_24h": 0.0,
                    })
                    poly_count += 1
            print(f"{poly_count} pts")
        else:
            print("0")
        time.sleep(0.3)

        # Kalshi trades -> hourly
        print(f"  Kalshi...", end=" ", flush=True)
        trades = fetch_kalshi_trades(pair["kalshi_ticker"])
        candles = kalshi_trades_to_hourly(trades)
        kalshi_count = 0
        if candles:
            for c in candles:
                all_records.append({
                    "timestamp": c["t"],
                    "market_id": pair["market_id"],
                    "platform": "kalshi",
                    "yes_price": c["p"],
                    "volume_24h": float(c.get("vol", 0)),
                })
                kalshi_count += 1
            print(f"{kalshi_count} pts (from {len(trades)} trades)")
        else:
            # Fallback: use current snapshot from market detail
            kalshi_last = pair.get("_kalshi_last")
            if kalshi_last is not None:
                snap_price = kalshi_last / 100.0 if kalshi_last > 1 else kalshi_last
                now_ts = int(time.time())
                now_hour = now_ts - (now_ts % 3600)
                all_records.append({
                    "timestamp": now_hour,
                    "market_id": pair["market_id"],
                    "platform": "kalshi",
                    "yes_price": round(snap_price, 4),
                    "volume_24h": 0.0,
                })
                kalshi_count = 1
                print(f"1 pt (snapshot: {snap_price:.2f})")
            else:
                print("0")
        time.sleep(0.3)

        if poly_count > 0 or kalshi_count > 0:
            successful += 1

    if not all_records:
        print("\nNo price data fetched.")
        return 1

    # Save
    all_records.sort(key=lambda r: (r["timestamp"], r["market_id"], r["platform"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PRICES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} price records to {PRICES_PATH}")

    # Clean pairs for output
    clean_pairs = []
    for p in top_pairs:
        clean_pairs.append({
            "market_id": p["market_id"],
            "polymarket_slug": p["polymarket_id"],
            "kalshi_ticker": p["kalshi_ticker"],
            "description": p["description"],
            "resolution_source": "",
        })

    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_pairs, f, indent=2)
    print(f"Saved {len(clean_pairs)} market pairs to {PAIRS_PATH}")
    print(f"\n{successful}/{len(top_pairs)} pairs have price data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
