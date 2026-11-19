"""測試輔助：只負責「構造輸入」，不負責計算預期值（預期值一律在各測試中人工寫死）。"""
import datetime as dt
import sys, os
from decimal import Decimal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uep_sim as u  # noqa: E402

TZ8 = dt.timezone(dt.timedelta(hours=8))


def weekdays(start: dt.date, n: int):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def candles_json(rows, symbol="TEST"):
    """rows: list of (date, O, H, L, C, volume)；輸出 fugle_candles 格式（新到舊）。turnover = 均價 × 量。"""
    data = []
    for d, o, h, l, c, v in sorted(rows, key=lambda r: r[0], reverse=True):
        avg = (Decimal(str(h)) + Decimal(str(l))) / 2
        data.append({"date": d.isoformat(), "open": o, "high": h, "low": l, "close": c,
                     "volume": v, "turnover": int(avg * v)})
    return {"queried_at": "x", "source": "test", "data": {"symbol": symbol, "timeframe": "D", "data": data, "sort": "desc"}}


def ratios_from_path(start_prev, days, ex=None):
    """days: list of (O,H,L,C) 的「目標價格」（元）；ex: {day_index: 每股股利}。
    回傳以前一日目標收盤（除息日用 前收 − 股利）為參考的比值陣列，使引擎重現目標價格。"""
    ex = ex or {}
    prev = Decimal(str(start_prev))
    r = []
    for i, (o, h, l, c) in enumerate(days):
        ref = prev - Decimal(str(ex.get(i, 0)))
        r.append([float(Decimal(str(x)) / ref) for x in (o, h, l, c)])
        prev = Decimal(str(c))
    return np.array(r, dtype=np.float64)


def one_path(days, start_prev, *, vol=10**9, V=None, D=None, **kw):
    """以 idx[0, d] = d 跑單一路徑；D 預設 = len(days)。"""
    ex = {}
    if "ex_days" in kw:
        for d, uu in zip(kw["ex_days"], kw["ex_div_u"]):      # 同日多筆要加總（不得覆蓋）
            ex[d] = ex.get(d, Decimal(0)) + Decimal(uu) / 10**6
    R = ratios_from_path(start_prev, days, ex)
    D = D or len(days)
    idx = np.arange(D, dtype=np.int32).reshape(1, D)
    vols = np.full(len(days), vol, np.int64) if np.isscalar(vol) else np.array(vol, np.int64)
    return u.run_core(R, vols, idx, V, start_prev_c=u.d2c(start_prev), trace_n=1, **kw)


def C(x):
    return u.d2c(x)


COSTS = {"commission_rate": "0.001425", "discount": "1", "min_fee_whole": "20", "min_fee_odd": "1",
         "tax_rate": "0.003", "fee_rounding": "ROUND_FLOOR", "tax_rounding": "ROUND_FLOOR",
         "fee_rounding_verified": False, "tax_rounding_verified": False}


def synthetic_market(n_days=70, seed=7, start=dt.date(2026, 8, 3), base=600.0, today=None):
    """合成的個股與 IX0001 日 K（合法價位），用於需要完整 run() 的測試。"""
    rng = np.random.default_rng(seed)
    dates = weekdays(start, n_days)
    rows, idx_rows = [], []
    prev = Decimal(str(base))
    for d in dates:
        dn, up = u.limits_dec(prev)
        c = u.round_legal_dec(prev * Decimal(str(float(np.exp(rng.normal(0, 0.03))))))
        c = min(max(c, dn), up)
        o = min(max(u.round_legal_dec(prev * Decimal(str(float(np.exp(rng.normal(0, 0.01)))))), dn), up)
        h = min(max(o, c, u.round_legal_dec(max(o, c) * Decimal("1.01"))), up)
        l = max(min(o, c, u.round_legal_dec(min(o, c) * Decimal("0.99"))), dn)
        rows.append((d, float(o), float(h), float(l), float(c), int(rng.integers(10_000_000, 30_000_000))))
        idx_rows.append((d, 20000.0, 20100.0, 19900.0, 20050.0, 5_000_000_000))
        prev = c
    return rows, idx_rows


def snapshot(last, bid, today, last_trade_dt, official_close=None, queried_dt=None):
    q = queried_dt or last_trade_dt
    return {"last_price": last, "best_bid": bid, "official_close": official_close,
            "last_trade_at": int(last_trade_dt.replace(tzinfo=TZ8).timestamp() * 1e6),
            "queried_at": q.strftime("%Y-%m-%d %H:%M:%S") + " Asia/Taipei"}


def holding_input(rows, idx_rows, today, *, phase="intraday", last=None, bid=None, tp=None, stop=None,
                  stop_type="close_confirm", shares=450, P=100_000, seed=1, events=None, cal=None,
                  official_close=None, last_trade=None, queried=None, Hd=20):
    last = last if last is not None else rows[-1][4]
    bid = bid if bid is not None else last
    lt = last_trade or dt.datetime.combine(today, dt.time(10, 0))
    inp = {"mode": "holding", "run_phase": phase, "price_limit_normal": True,
           "raw_candles": candles_json(rows), "raw_index_candles": candles_json(idx_rows, "IX0001"),
           "market_snapshot": snapshot(last, bid, today, lt, official_close, queried),
           "plan": {"shares": shares, "take_profit": tp, "stop_trigger": stop, "stop_type": stop_type, "Hd": Hd},
           "costs": dict(COSTS), "params": {"path_count": P, "seed": seed, "extension_cap": 10, "freshness_minutes": 5}}
    if events is not None:
        inp["future_events"] = events
    if cal is not None:
        inp["trading_calendar"] = cal
    return inp


def prices_of(trace):
    """trace（decode_trace 輸出）→ {day: (O,H,L,C,DN,UP)}，Decimal 元。"""
    return {e["day"]: (e["O"], e["H"], e["L"], e["C"], e["DN"], e["UP"]) for e in trace if e["event"] == "PRICES"}


def events(trace, name):
    return [e for e in trace if e["event"] == name]


def check_path_built(out, days):
    """構造自檢：引擎重現的 OHLC 必須等於測試指定的目標價（避免測試本身構造錯誤）。"""
    px = prices_of(out["trace"][0])
    for d, row in px.items():
        want = tuple(Decimal(str(x)) for x in days[d])
        assert row[:4] == want, f"day {d}: engine {row[:4]} != target {want}"


def mk_out(cls, exit_px, shares, *, cons_px=None, div=None, sub_px=None):
    """手動構造 simulate_core 的輸出（只用於測試評估層 §5）。價格以元給定。"""
    P = len(cls)
    return {"cls": np.array(cls, np.int64), "exit_px": np.array([u.d2c(x) for x in exit_px], np.int64),
            "cons_px": np.array([u.d2c(x) for x in (cons_px or [0] * P)], np.int64),
            "shares": np.array(shares, np.int64), "div": np.array(div or [0] * P, np.int64),
            "sub_px": np.array(sub_px if sub_px is not None else [[0, 0, 0, 0]] * P, np.int64),
            "sub_kind": np.zeros((P, 4), np.int64), "delay": np.zeros(P, np.int64),
            "dev_sum": np.zeros(P), "dev_n": np.ones(P, np.int64), "clip_n": np.zeros(P, np.int64),
            "lock_n": np.zeros(P, np.int64)}


def Vpath(D, cols):
    """單路徑 V：cols = {col: [每日值]}，其餘為 0.99。"""
    V = np.full((1, D, 4), 0.99)
    for c, vals in cols.items():
        V[0, :len(vals), c] = vals
    return V
