import asyncio
import csv
import datetime as dt
import io
import os
import zipfile
from typing import Iterable, List, Dict, Any

import httpx

BINANCE_VISION_BASE = "https://data.binance.vision"
BINANCE_FUTURES_API_BASE = "https://fapi.binance.com"
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


# -----------------------
# Utility helpers
# -----------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def to_epoch_millis_utc(date: dt.date) -> int:
    dt_obj = dt.datetime(
        date.year,
        date.month,
        date.day,
        tzinfo=dt.timezone.utc,
    )
    return int(dt_obj.timestamp() * 1000)


def day_bounds_utc(date: dt.date) -> (int, int):
    start_dt = dt.datetime(
        date.year,
        date.month,
        date.day,
        tzinfo=dt.timezone.utc,
    )
    end_dt = start_dt + dt.timedelta(days=1) - dt.timedelta(milliseconds=1)
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)


# -----------------------
# Binance Vision klines
# -----------------------


def binance_kline_url(
        market: str,
        symbol: str,
        interval: str,
        date: dt.date,
) -> str:
    """
    market: "spot" or "um_futures"
    symbol: e.g. "BTCUSDT"
    interval: e.g. "1s", "1m"
    """
    date_str = date.strftime("%Y-%m-%d")
    if market == "spot":
        return (
            f"{BINANCE_VISION_BASE}/data/spot/daily/klines/"
            f"{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
        )
    elif market == "um_futures":
        return (
            f"{BINANCE_VISION_BASE}/data/futures/um/daily/klines/"
            f"{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
        )
    else:
        raise ValueError(f"Unknown market: {market}")


async def fetch_and_extract_binance_zip(
        client: httpx.AsyncClient,
        url: str,
        out_dir: str,
) -> None:
    print(f"[BINANCE] downloading {url}")
    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            print(f"[BINANCE] 404 for {url}")
            return
        resp.raise_for_status()
    except Exception as exc:
        print(f"[BINANCE] failed {url}: {exc}")
        return

    ensure_dir(out_dir)

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                dest_path = os.path.join(out_dir, os.path.basename(member))
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                print(f"[BINANCE] saved {dest_path}")
    except Exception as exc:
        print(f"[BINANCE] unzip error for {url}: {exc}")


async def fetch_binance_klines_range(
        client: httpx.AsyncClient,
        symbols: List[str],
        start_date: dt.date,
        end_date: dt.date,
        base_out_dir: str,
        spot_interval: str = "1s",
        futures_interval: str = "1m",
) -> None:
    print(
        f"[BINANCE] fetching klines "
        f"symbols={symbols}, dates={start_date}..{end_date}, "
        f"spot={spot_interval}, futures={futures_interval}"
    )

    tasks = []

    for symbol in symbols:
        for d in daterange(start_date, end_date):
            # Spot
            spot_url = binance_kline_url("spot", symbol, spot_interval, d)
            spot_out = os.path.join(
                base_out_dir,
                "binance",
                "spot",
                symbol,
                spot_interval,
            )
            tasks.append(
                fetch_and_extract_binance_zip(client, spot_url, spot_out)
            )


    # USDT-margined futures (perp)
    fut_url = binance_kline_url("um_futures", symbol, futures_interval, d)
    fut_out = os.path.join(
        base_out_dir,
        "binance",
        "futures_um",
        symbol,
        futures_interval,
    )
    tasks.append(
        fetch_and_extract_binance_zip(client, fut_url, fut_out)
    )

    await asyncio.gather(*tasks)
    print("[BINANCE] klines done")


# -----------------------
# Binance futures aggTrades (tick-level)
# -----------------------


async def fetch_binance_futures_agg_trades_for_day(
        client: httpx.AsyncClient,
        symbol: str,
        date: dt.date,
) -> List[Dict[str, Any]]:
    """
    Fetch all aggTrades for a futures symbol during given UTC calendar date.
    Uses short pauses between requests and retry on HTTP 429.
    """
    start_ms, end_ms = day_bounds_utc(date)
    all_trades: List[Dict[str, Any]] = []

    last_id: int | None = None
    done = False
    batch_index = 0
    consecutive_429 = 0
    max_429_retries = 10

    print(
        f"[BINANCE aggTrades] start symbol={symbol}, "
        f"date={date.isoformat()}, window=[{start_ms}, {end_ms}]"
    )

    while not done:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "limit": 1000,
        }
        if last_id is None:
            params["startTime"] = start_ms
        else:
            params["fromId"] = last_id + 1

        try:
            resp = await client.get(
                f"{BINANCE_FUTURES_API_BASE}/fapi/v1/aggTrades",
                params=params,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                consecutive_429 += 1
                print(
                    f"[BINANCE aggTrades] 429 for {symbol} "
                    f"(batch={batch_index}, last_id={last_id}), "
                    f"retry #{consecutive_429} after short sleep"
                )
                if consecutive_429 > max_429_retries:
                    print(
                        f"[BINANCE aggTrades] too many 429 for {symbol}, "
                        f"giving up for this day"
                    )
                    break
                await asyncio.sleep(1.5)
                continue
            print(f"[BINANCE aggTrades] HTTP error for {symbol}: {exc}")
            break
        except Exception as exc:
            print(f"[BINANCE aggTrades] failed for {symbol}: {exc}")
            break

        consecutive_429 = 0
        trades = resp.json()
        if not trades:
            print(
                f"[BINANCE aggTrades] empty response for {symbol}, "
                f"batch={batch_index}, stop"
            )
            break

        batch_index += 1
        batch_len = len(trades)
        last_ts = int(trades[-1]["T"])
        last_trade_id = int(trades[-1]["a"])

        print(
            f"[BINANCE aggTrades] {symbol} {date.isoformat()} "
            f"batch={batch_index}, size={batch_len}, "
            f"last_id={last_trade_id}, last_ts={last_ts}"
        )

        for t in trades:
            ts = int(t["T"])
            trade_id = int(t["a"])
            if ts > end_ms:
                done = True
                break
            all_trades.append(t)
            last_id = trade_id

        if len(trades) < 1000:
            print(
                f"[BINANCE aggTrades] {symbol} {date.isoformat()} "
                f"got <1000 trades in batch, stop"
            )
            break

        # shorter pause between batches, just to be nice with rate limits
        await asyncio.sleep(0.5)

    print(
        f"[BINANCE aggTrades] collected {len(all_trades)} trades "
        f"for {symbol} on {date.isoformat()}"
    )
    return all_trades


def save_agg_trades_csv(
        trades: List[Dict[str, Any]],
        out_path: str,
) -> None:
    if not trades:
        print(f"[BINANCE aggTrades] no trades, skip {out_path}")
        return

    ensure_dir(os.path.dirname(out_path))

    fieldnames = ["a", "p", "q", "f", "l", "T", "m", "M"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            row = {k: t.get(k) for k in fieldnames}
            writer.writerow(row)

    print(f"[BINANCE aggTrades] saved {out_path}")


async def fetch_binance_futures_trades_range(
        client: httpx.AsyncClient,
        symbols: List[str],
        start_date: dt.date,
        end_date: dt.date,
        base_out_dir: str,
) -> None:
    print(
        f"[BINANCE aggTrades] fetching range "
        f"symbols={symbols}, dates={start_date}..{end_date}"
    )
    for symbol in symbols:
        for d in daterange(start_date, end_date):
            print(
                f"[BINANCE aggTrades] fetching trades for "
                f"{symbol} {d.isoformat()}"
            )
            trades = await fetch_binance_futures_agg_trades_for_day(
                client,
                symbol=symbol,
                date=d,
            )
            filename = f"aggTrades_{symbol}_{d.isoformat()}.csv"
            out_path = os.path.join(
                base_out_dir,
                "binance",
                "futures_um_trades",
                symbol,
                filename,
            )
            save_agg_trades_csv(trades, out_path)
    print("[BINANCE aggTrades] done")


# -----------------------
# Hyperliquid candles
# -----------------------


async def fetch_hyperliquid_candles(
        client: httpx.AsyncClient,
        coin: str,
        interval: str,
        start_ms: int,
        end_ms: int,
) -> List[Dict[str, Any]]:
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    print(
        f"[HL] request coin={coin}, interval={interval}, "
        f"window=[{start_ms}, {end_ms}]"
    )
    resp = await client.post(HYPERLIQUID_INFO_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Hyperliquid response for {coin}: {data}")
    print(f"[HL] got {len(data)} candles for {coin}")
    return data


def save_hyperliquid_csv(
        candles: List[Dict[str, Any]],
        out_path: str,
) -> None:
    if not candles:
        print(f"[HL] no candles for {out_path}")
        return

    ensure_dir(os.path.dirname(out_path))

    fieldnames = list(candles[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in candles:
            writer.writerow(row)

    print(f"[HL] saved {out_path}")


async def fetch_hyperliquid_range(
        client: httpx.AsyncClient,
        coins: List[str],
        interval: str,
        start_date: dt.date,
        end_date: dt.date,
        base_out_dir: str,
) -> None:
    print(
        f"[HL] fetching candles coins={coins}, "
        f"interval={interval}, dates={start_date}..{end_date}"
    )

    start_ms = to_epoch_millis_utc(start_date)
    end_ms = to_epoch_millis_utc(end_date + dt.timedelta(days=1)) - 1

    for coin in coins:
        try:
            candles = await fetch_hyperliquid_candles(
                client,
                coin=coin,
                interval=interval,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        except Exception as exc:
            print(f"[HL] failed for {coin}: {exc}")
            continue

        filename = (
            f"{coin}_{interval}_"
            f"{start_date.isoformat()}_{end_date.isoformat()}.csv"
        )
        out_path = os.path.join(
            base_out_dir,
            "hyperliquid",
            "perp",
            coin,
            interval,
            filename,
        )
        save_hyperliquid_csv(candles, out_path)

    print("[HL] done")


# -----------------------
# Main
# -----------------------

async def main() -> None:
    # Crash date
    target_date = dt.date(2025, 10, 10)
    start_date = target_date
    end_date = target_date

    # Binance symbols (spot + UM futures)
    binance_symbols = ["BTCUSDT", "ETHUSDT"]

    # Hyperliquid perp coins
    hyperliquid_coins = ["BTC", "ETH"]
    hyperliquid_interval = "15m"

    base_out_dir = "data"

    print("[MAIN] starting")
    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        print("[MAIN] step 1: Binance klines")
        await fetch_binance_klines_range(
            client,
            symbols=binance_symbols,
            start_date=start_date,
            end_date=end_date,
            base_out_dir=base_out_dir,
            spot_interval="1s",
            futures_interval="1m",
        )

        print("[MAIN] step 2: Binance futures aggTrades")
        await fetch_binance_futures_trades_range(
            client,
            symbols=binance_symbols,
            start_date=start_date,
            end_date=end_date,
            base_out_dir=base_out_dir,
        )

        print("[MAIN] step 3: Hyperliquid candles")
        await fetch_hyperliquid_range(
            client,
            coins=hyperliquid_coins,
            interval=hyperliquid_interval,
            start_date=start_date,
            end_date=end_date,
            base_out_dir=base_out_dir,
        )

    print("[MAIN] done")


if __name__ == "__main__":
    asyncio.run(main())
