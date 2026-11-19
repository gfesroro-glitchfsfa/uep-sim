"""需要完整 run()（§1～§5 全流程）的 §7 測試：8、18、19、21、29/45、30（V 雜湊）、43、44（欄位清單）、47、63、64、67。
使用合成日 K（合法價位、無缺日）；路徑數為規格下限 100,000。"""
import copy
import datetime as dt
import hashlib
import json
from decimal import Decimal as Dm

import numpy as np
import pytest

from helpers import u, synthetic_market, holding_input, candles_json, snapshot, COSTS, weekdays

P = 100_000
SEED = 20261119


def next_weekday(d):
    d += dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def legal(x):
    return float(u.round_legal_dec(Dm(str(x))))


@pytest.fixture(scope="module")
def mkt70():
    rows, idx = synthetic_market(70, seed=7)
    return rows, idx, next_weekday(rows[-1][0])


def hold_inp(rows, idx, today, **kw):
    last = kw.pop("last", rows[-1][4])
    base = dict(last=last, bid=600.0, tp=legal(last * 1.15), stop=legal(last * 0.9), shares=450, P=P, seed=SEED, Hd=3,
                last_trade=dt.datetime.combine(today, dt.time(10, 0)), queried=dt.datetime.combine(today, dt.time(10, 7)))
    base.update(kw)
    return holding_input(rows, idx, today, **base)


@pytest.fixture(scope="module")
def intraday_run(mkt70):
    rows, idx, today = mkt70
    inp = hold_inp(rows, idx, today)
    return inp, u.run(copy.deepcopy(inp)), u.run(copy.deepcopy(inp))


# ---------------------------------------------------------------- 8
def test_08_common_sampling_hashes(intraday_run, mkt70):
    inp, r, _ = intraday_run
    rows = mkt70[0]
    core = r["core"]
    D = 0 + 3 + 10 + 1
    U = np.random.Generator(np.random.PCG64(SEED)).random((P, D), dtype=np.float64)   # 獨立重建
    assert core["uniform_U_sha256"] == hashlib.sha256(U.astype("<f8").tobytes()).hexdigest()
    assert core["uniform_V_sha256"] is None                                        # holding 不產生 V
    sample_dates = [r_[0] for r_ in rows][1:]                                      # 合成資料無缺日：樣本日 = 第 2 根起
    hs = {}
    for w in (20, 40, 60):
        wm = core["windows"][str(w)]
        idx = np.floor(U * w).astype(np.int32)
        assert wm["sampling_indices_sha256"] == hashlib.sha256(idx.astype("<i4").tobytes()).hexdigest()
        dates = [d.isoformat() for d in sample_dates[-w:]]
        joined = "\n".join(dates[i] for i in idx.ravel())
        assert wm["sampling_dates_sha256"] == hashlib.sha256(joined.encode("utf-8")).hexdigest()
        assert (wm["seed"], wm["path_count"], wm["sample_last"]) == (SEED, P, dates[-1])
        hs[w] = wm["sampling_dates_sha256"]
    assert len(set(hs.values())) == 3                                               # 不同窗：日期不同
    # 同窗 5 個漂移版本：共用同一組 idx（版本清單依窗分組、每窗 5 個 f）
    assert [(v["window"], v["drift_keep"]) for v in core["versions"]] == [(w, f) for w in (20, 40, 60) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]


# ---------------------------------------------------------------- 18
def test_18_reproducible_core_hash(intraday_run, mkt70):
    inp, r1, r2 = intraday_run
    assert r1["core_sha256"] == r2["core_sha256"]
    canon = json.dumps(r1["core"], sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    assert r1["core_sha256"] == hashlib.sha256(canon).hexdigest()
    assert "run_at" not in json.dumps(r1["core"], ensure_ascii=False)               # meta 欄位不在 core
    m = copy.deepcopy(r1)
    m["meta"]["run_at"] = "1999-01-01T00:00:00"
    assert hashlib.sha256(json.dumps(m["core"], sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                     default=str).encode()).hexdigest() == r1["core_sha256"]
    rows, idx, today = mkt70
    r3 = u.run(hold_inp(rows, idx, today, seed=SEED + 2))
    assert r3["core_sha256"] != r1["core_sha256"]


# ---------------------------------------------------------------- 21 / 64 / 67（intraday 部分）
def test_21_64_67_intraday_start_flags_and_B(intraday_run):
    inp, r, _ = intraday_run
    core = r["core"]
    assert core["start_prev"] == inp["market_snapshot"]["last_price"]
    assert "intraday：隔日漲跌停價為估算，非正式基準" in core["flags"]
    assert any(f.startswith("最新成交距查詢已超過 7.0 分鐘") for f in core["flags"])
    assert "費用取整尚未核實，淨期望值含未確認的成本假設" in core["flags"]
    assert core["baseline_B"] == "268806"          # 600 × 450 = 270000 − floor(384.75) − floor(810)
    assert core["costs"]["fee_rounding_verified"] is False
    assert core["tax_note"].startswith("本引擎損益已扣除交易手續費、證交稅及模型設定之滑價，但未扣除個人股利所得稅及二代健保補充保費。")
    assert len(core["versions"]) == 15 and core["windows_skipped"] == {}
    for v in core["versions"]:
        assert v["result"]["prob_sum"] == pytest.approx(1.0, abs=1e-6)


def test_21_63_after_close_uses_official_close_and_includes_today(mkt70):
    rows, idx, today = mkt70
    c_today = legal(rows[-1][4] * 1.01)
    rows2 = rows + [(today, rows[-1][4], max(c_today, rows[-1][4]), min(c_today, rows[-1][4]), c_today, 12_000_000)]
    idx2 = idx + [(today, 20000.0, 20100.0, 19900.0, 20050.0, 5_000_000_000)]
    inp = hold_inp(rows2, idx2, today, phase="after_close", last=c_today, official_close=c_today,
                   last_trade=dt.datetime.combine(today, dt.time(13, 30)), queried=dt.datetime.combine(today, dt.time(14, 0)))
    r = u.run(inp)
    core = r["core"]
    assert core["start_prev"] == c_today and core["data_log"]["after_close_today_included"] is True
    assert core["usable_sample_days"] == 70 and core["windows"]["20"]["sample_last"] == today.isoformat()
    assert not any(("intraday" in f) or ("分鐘" in f) for f in core["flags"])
    bad = copy.deepcopy(inp)
    bad["market_snapshot"]["official_close"] = legal(c_today + 1)
    with pytest.raises(u.EngineRefusal, match="官方收盤價"):
        u.run(bad)


# ---------------------------------------------------------------- 64
def test_64_freshness_refusals(mkt70):
    rows, idx, today = mkt70
    inp = hold_inp(rows, idx, today)
    a = copy.deepcopy(inp)
    a["market_snapshot"]["last_trade_at"] = None
    with pytest.raises(u.EngineRefusal, match="last_trade_at"):
        u.run(a)
    for phase in ("intraday", "after_close"):
        b = hold_inp(rows, idx, today, phase=phase, official_close=rows[-1][4],
                     last_trade=dt.datetime.combine(today - dt.timedelta(days=1), dt.time(13, 30)))
        with pytest.raises(u.EngineRefusal, match="今天沒有成交"):
            u.run(b)


# ---------------------------------------------------------------- 19 / 43
def test_19_43_45_days_runs_10_versions():
    rows, idx = synthetic_market(46, seed=9)
    today = next_weekday(rows[-1][0])
    r = u.run(hold_inp(rows, idx, today))
    core = r["core"]
    assert core["usable_sample_days"] == 45 and core["windows_run"] == [20, 40]
    assert core["windows_skipped"] == {"60": "無法完成（只有 45 日）"} and len(core["versions"]) == 10


def test_19_refusals_short_data_and_no_price_limit():
    rows, idx = synthetic_market(16, seed=9)                      # 15 個可用日
    today = next_weekday(rows[-1][0])
    with pytest.raises(u.EngineRefusal, match="不足 20 日"):
        u.run(hold_inp(rows, idx, today))
    rows, idx = synthetic_market(30, seed=9)
    today = next_weekday(rows[-1][0])
    inp = hold_inp(rows, idx, today)
    inp["price_limit_normal"] = False
    with pytest.raises(u.EngineRefusal, match="無漲跌幅限制"):
        u.run(inp)


# ---------------------------------------------------------------- 29 / 45（經由 run）
@pytest.mark.parametrize("typ,msg", [("stock_dividend", "股票股利"), ("capital_reduction", "減資")])
def test_29_45_run_refuses_events_in_horizon(mkt70, typ, msg):
    rows, idx, today = mkt70
    cal = [d.isoformat() for d in weekdays(next_weekday(today), 20)]
    inp = hold_inp(rows, idx, today, events=[{"type": typ, "ex_date": cal[3]}], cal=cal)
    with pytest.raises(u.EngineRefusal, match=msg):
        u.run(inp)


# ---------------------------------------------------------------- 47
def entry_inp(rows, idx, today, **kw):
    last = rows[-1][4]
    inp = holding_input(rows, idx, today, last=last, bid=last, P=P, seed=SEED,
                        tp=kw.pop("tp", legal(last * 1.15)), stop=kw.pop("stop", legal(last * 0.88)), Hd=3,
                        last_trade=dt.datetime.combine(today, dt.time(10, 0)), queried=dt.datetime.combine(today, dt.time(10, 1)))
    inp["mode"] = "entry"
    inp["plan"].update({"E": 2, "batches": kw.pop("batches", [{"limit": legal(last * 0.98), "shares": 1000},
                                                                {"limit": legal(last * 0.95), "shares": 300}]),
                        "batch2_activation": "open"})
    inp["fill_interval"] = kw.pop("fill_interval", [0.3, 0.7])
    return inp


def test_47_input_checks(mkt70):
    rows, idx, today = mkt70
    last = rows[-1][4]
    with pytest.raises(u.EngineRefusal, match="不是合法價位"):
        u.run(hold_inp(rows, idx, today, tp=round(legal(last * 1.15) + 0.01, 2)))           # 價格 ≥ 10 時 0.01 不是升降單位
    with pytest.raises(u.EngineRefusal, match="停利價"):
        u.run(hold_inp(rows, idx, today, tp=last))                                       # 停利 ≤ 起始價
    with pytest.raises(u.EngineRefusal, match="entry"):
        u.run(entry_inp(rows, idx, today, batches=[{"limit": legal(last * 1.2), "shares": 1000}]))   # 限價 ≥ 停利
    with pytest.raises(u.EngineRefusal, match="fill_interval"):
        u.run(entry_inp(rows, idx, today, fill_interval=[0.7, 0.3]))


# ---------------------------------------------------------------- 30 / 43 / 44 / 67（entry 全流程）
def test_30_43_44_67_entry_run():
    rows, idx = synthetic_market(26, seed=13)                     # 25 個可用日 → 5 個版本
    today = next_weekday(rows[-1][0])
    r = u.run(entry_inp(rows, idx, today))
    core = r["core"]
    assert core["windows_run"] == [20] and len(core["versions"]) == 5
    assert core["windows_skipped"] == {"40": "無法完成（只有 25 日）", "60": "無法完成（只有 25 日）"}
    D = 2 + 3 + 10 + 1
    V = np.random.Generator(np.random.PCG64(SEED + 1)).random((P, D, 4), dtype=np.float64)
    assert core["uniform_V_sha256"] == hashlib.sha256(V.astype("<f8").tobytes()).hexdigest()
    assert core["V_column_map"] == {"0": "batch1_whole", "1": "batch1_odd", "2": "batch2_whole", "3": "batch2_odd"}
    assert [(s["col"], s["batch"], s["type"], s["shares"]) for s in core["sub_orders_present"]] == [(0, 1, "whole", 1000), (3, 2, "odd", 300)]
    assert "費用取整尚未核實，淨期望值含未確認的成本假設" in core["flags"]
    for v in core["versions"]:
        lo, hi = v["fill_low"], v["fill_high"]
        assert lo["p_any_fill"] <= hi["p_any_fill"]               # 同一個 V，p 越大成交越多
        for x in (lo, hi):
            assert x["prob_sum"] == pytest.approx(1.0, abs=1e-6)
            assert set(x["sub_orders"]) == {"batch1_whole", "batch2_odd"}
    assert "只評估第一批全部成交" in core["deterministic_stress"]["note"]
