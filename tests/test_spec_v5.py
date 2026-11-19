"""uep_sim_spec_v5.md §7 測試 1～67（引擎核心與評估層）。
預期值一律為人工推導的常數（推導寫在註解），不呼叫引擎函數產生預期值。
需要完整 run() 的項目（8、18、19 部分、21、43 部分、47、63、64、67 部分）放在 test_integration_run.py。
價格以元表示；costs 一律無折扣、整股最低 20、零股最低 1、FLOOR（verified=false）。"""
from decimal import Decimal as Dm
import math
import numpy as np
import pytest

from helpers import u, one_path, C, COSTS, check_path_built, events, prices_of, mk_out, Vpath

COST = u.Costs(COSTS)
TP, SL, CL, EXP, UNF, NOF = u.CLS_TP, u.CLS_SL, u.CLS_CLOSE, u.CLS_EXP, u.CLS_UNF, u.CLS_NOFILL
INTRA, CLOSE = 0, 1


def hold(days, prev, *, tp, stop, st, Hd, shares=1000, **kw):
    o = one_path(days, prev, mode=0, hold_shares=shares, tp_c=C(tp), stop_c=C(stop), stop_type=st, Hd=Hd, **kw)
    check_path_built(o, days)
    return o


def entry(days, prev, *, tp, stop, st, Hd, subs, lp, E, after=0, fill_p=0.5, V=None, vol=10**9, **kw):
    D = len(days)
    V = V if V is not None else np.full((1, D, 4), 0.99)
    o = one_path(days, prev, mode=1, tp_c=C(tp), stop_c=C(stop), stop_type=st, Hd=Hd, sub_shares=subs,
                 lp_c=(C(lp[0]), C(lp[1]) if len(lp) > 1 else 0), E=E, batch2_after=after, fill_p=fill_p, V=V, vol=vol, **kw)
    check_path_built(o, days)
    return o


def net(o, i=0):
    """單路徑 Vᵢ（元，整數 1e-6 元精確換算）。"""
    return Dm(int(o["_V_main_u"][i])) / 10**6


# ---------------------------------------------------------------- 1
def test_01_lock_delays_exit_one_day():
    # prev 100 → DN 90；d0 一價到底 90 且 L ≤ 停損 95（盤中）→ 待執行；d1 prev 90 → DN 81
    # d1 O 85 → max(tick_down(85)=84.9, L 84, DN 81) = 84.9；順延 1 日
    o = hold([(90, 90, 90, 90), (85, 88, 84, 86)], 100, tp=110, stop=95, st=INTRA, Hd=5)
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0], o["delay"][0]) == (SL, 1, 8490, 1)
    u.evaluate_version(o, COST, 0, B=COST.sell_net(10000, 1000))
    assert net(o) == 84526  # 84900 − 120 − 254（延遲期間的跌價照算，以 d1 成交價計）


# ---------------------------------------------------------------- 2
def test_02_open_at_limit_down_then_opens_fills_at_DN():
    # 收盤確認：d0 C 94 < 95 → 待執行；d1 prev 94 → DN 84.6；O = DN 但盤中打開（非 LOCK）
    o = hold([(99, 99, 94, 94), (84.6, 88, 84.6, 87)], 100, tp=110, stop=95, st=CLOSE, Hd=5)
    assert (o["cls"][0], o["exit_px"][0], o["delay"][0]) == (CL, 8460, 0)  # max(84.5, 84.6, 84.6)
    # 盤中型：開盤 90 ＝ DN 低於停損 → 基準 O → max(89.9, 90, 90) = 90
    o2 = hold([(90, 93, 90, 92)], 100, tp=110, stop=95, st=INTRA, Hd=5)
    assert (o2["cls"][0], o2["exit_px"][0]) == (SL, 9000)


# ---------------------------------------------------------------- 3 / 4
def test_03_close_confirm_tp_through_and_close_below_is_TP():
    o = hold([(100, 106, 94, 94)], 100, tp=105, stop=95, st=CLOSE, Hd=5)
    assert (o["cls"][0], o["exit_px"][0]) == (TP, 10500)
    assert not events(o["trace"][0], "PENDING")


def test_04_intraday_tp_through_and_low_below_stop_is_SL():
    o = hold([(100, 106, 94, 94)], 100, tp=105, stop=95, st=INTRA, Hd=5)
    assert (o["cls"][0], o["exit_px"][0]) == (SL, 9490)  # base 95 → max(94.9, 94, 90)


# ---------------------------------------------------------------- 5
def test_05_trigger_on_Hd_fills_next_open_and_lock_delays_again():
    # Hd=3：d2 為第 Hd 日，C 94 < 95 → 待執行（不另做期滿）；d3 prev 94 LOCK 84.6；d4 prev 84.6 → DN 76.2
    days = [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 94, 94), (84.6, 84.6, 84.6, 84.6), (80, 81, 79, 80)]
    o = hold(days, 100, tp=110, stop=95, st=CLOSE, Hd=3)
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0], o["delay"][0]) == (CL, 4, 7990, 1)
    seq = [(e["day"], e["event"]) for e in o["trace"][0] if e["event"] in ("PENDING", "LOCK", "EXIT")]
    assert seq == [(2, "PENDING"), (3, "LOCK"), (4, "EXIT")]
    # 第 Hd 日 LOCK → 期滿轉待執行 → 次日以 max(tick_down(O), L, DN) 成交
    o2 = hold([(100,) * 4, (100,) * 4, (90,) * 4, (85, 88, 84, 86)], 100, tp=110, stop=80, st=CLOSE, Hd=3)
    assert (o2["cls"][0], o2["exit_day"][0], o2["exit_px"][0], o2["delay"][0]) == (EXP, 3, 8490, 1)


# ---------------------------------------------------------------- 6
LOCK_CHAIN_FROM_90 = [81, 72.9, 65.7, 59.2, 53.3, 48.0, 43.2, 38.9, 35.05, 31.55]  # 逐日 ceil_legal(前收×0.9)，人工計算


def test_06_extension_cap_unfinished_main_and_conservative():
    days = [(90,) * 4] + [(x,) * 4 for x in LOCK_CHAIN_FROM_90]   # d0 為第 Hd 日（Hd=1）一價到底
    o = hold(days, 100, tp=110, stop=80, st=CLOSE, Hd=1)
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0], o["cons_px"][0]) == (UNF, 10, 3155, 2840)
    assert o["delay"][0] == 11  # d0～d10 共 11 個 LOCK 日
    r = u.evaluate_version(o, COST, 0, B=COST.sell_net(10000, 1000))
    # main：31550 − 44 − 94 = 31412；cons：28400 − 40 − 85 = 28275；B = 100000 − 142 − 300 = 99558
    assert r["ev"] == -68146 and r["ev_conservative_unfinished"] == -71283
    assert r["es"] == -68146 and r["es_conservative_unfinished"] == -71283
    assert r["unfinished_paths"] == 1


# ---------------------------------------------------------------- 7
def test_07_constant_price_zero_drift_ev_zero_and_bid_difference():
    win = [{"ratios": [1.0, 1.0, 1.0, 1.0]} for _ in range(20)]
    R = u.drift_ratios(win, 0.0, 0.0)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, 20, size=(50, 6))
    o = u.run_core(R, np.full(20, 10**9), idx, None, mode=0, start_prev_c=10000, hold_shares=1000, tp_c=11000,
                   stop_c=9000, stop_type=CLOSE, Hd=5)
    assert set(o["cls"].tolist()) == {EXP} and set(o["exit_px"].tolist()) == {10000}  # max(99.9, L 100, DN 90)
    assert u.evaluate_version(o, COST, 0, B=COST.sell_net(10000, 1000))["ev"] == 0
    # bid 99.9：B = 99900 − 142 − 299 = 99459；V = 99558 → EV = 99
    assert u.evaluate_version(o, COST, 0, B=COST.sell_net(9990, 1000))["ev"] == 99


# ---------------------------------------------------------------- 9
def test_09_sample_ex_dividend_ratio_uses_reference_price():
    import datetime as dt
    d = [dt.date(2026, 10, i) for i in (1, 2, 5)]
    rows = [(d[0], Dm(100), Dm(101), Dm(99), Dm(100), 1000, None),
            (d[1], Dm(96), Dm(97), Dm(95), Dm(96), 1000, None),
            (d[2], Dm(95), Dm(96), Dm(94), Dm(95), 1000, None)]
    idx = [(x, Dm(1), Dm(1), Dm(1), Dm(1), 1, None) for x in d]
    s, log = u.build_samples(rows, idx, "after_close", dt.date(2026, 10, 6), None, {d[1]: Dm("95")})
    assert s[0]["prev_ref"] == Dm(95) and s[0]["ratios"] == [96 / 95, 97 / 95, 1.0, 96 / 95]
    s2, log2 = u.build_samples(rows, idx, "after_close", dt.date(2026, 10, 6), None, {d[1]: None})
    assert [x["date"] for x in s2] == [d[2]] and log2["excluded_no_ref"] == [str(d[1])]


# ---------------------------------------------------------------- 10
def test_10_limits_match_calc_limit_prices_and_simulated_prices_legal():
    import json, os
    fx = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "calc_limit_prices_2026-11-19.json")))
    for c in fx["cases"]:
        p = Dm(c["prev"])
        for k in range(3):                          # 連續漲停路徑
            dn, up = u.limits_dec(p)
            assert up == Dm(c["up"][k]) and dn == Dm(c["up_path_lows"][k]), (c["prev"], k)
            dm, um = u.limits_micro(u.d2c(p) * 10_000)
            assert (dm, um) == (u.d2c(dn), u.d2c(up))
            p = up
        p = Dm(c["prev"])
        for k in range(3):                          # 連續跌停路徑
            dn, up = u.limits_dec(p)
            assert dn == Dm(c["dn"][k]) and up == Dm(c["dn_path_highs"][k]), (c["prev"], k)
            p = dn
    # 隨機路徑：每一個模擬價格合法、在 [DN, UP] 內、K 線不等式成立
    rng = np.random.default_rng(11)
    R = np.exp(rng.normal(0, 0.06, size=(60, 4)))
    idx = rng.integers(0, 60, size=(40, 12))
    for start in (995, 4990, 9990, 49950, 99900, 100500):
        o = u.run_core(R, np.full(60, 10**9), idx, None, mode=0, start_prev_c=start, hold_shares=1000,
                       tp_c=start * 3, stop_c=1, stop_type=CLOSE, Hd=11, trace_n=40)
        for tr in o["trace"]:
            for (O, H, L, Cc, DN, UP) in prices_of(tr).values():
                for x in (O, H, L, Cc, DN, UP):
                    assert u.is_legal_dec(x)
                assert DN <= L <= min(O, Cc) and H >= max(O, Cc) and H <= UP


# ---------------------------------------------------------------- 11
@pytest.mark.parametrize("bar,expect", [
    ((595, 606, 594, 600), (59500, u.K_GAP)),       # O < Lp → O
    ((605, 606, 599, 600), (60000, u.K_THROUGH)),   # L ≤ tick_down(600)=599 → Lp
    ((605, 606, 600, 602), (60000, u.K_TOUCH_V)),   # L = Lp，V=0.1 < 0.5 → Lp
    ((600, 606, 600, 602), (60000, u.K_TOUCH_V)),   # O = Lp 且未再穿過
    ((605, 606, 601, 602), (0, u.K_NEVER_TOUCHED)),  # 未觸及
])
def test_11a_buy_limit(bar, expect):
    o = entry([bar], 610, tp=650, stop=560, st=CLOSE, Hd=5, subs=(1000, 0, 0, 0), lp=(600,), E=1,
              V=Vpath(1, {0: [0.1]}))
    assert (o["sub_px"][0, 0], o["sub_kind"][0, 0]) == expect


def test_11b_buy_exact_touch_V_not_below_p_no_fill():
    o = entry([(605, 606, 600, 602)], 610, tp=650, stop=560, st=CLOSE, Hd=5, subs=(1000, 0, 0, 0), lp=(600,), E=1,
              V=Vpath(1, {0: [0.5]}), fill_p=0.5)                 # V < p 才成交；V = p 不成交
    assert (o["sub_px"][0, 0], o["sub_kind"][0, 0], o["cls"][0]) == (0, u.K_TOUCHED_NOFILL, NOF)


@pytest.mark.parametrize("bar,expect", [
    ((106, 107, 105, 106), (TP, 10600)),   # O > Tp → O
    ((104, 105.5, 103, 104), (TP, 10500)),  # H ≥ tick_up(105)=105.5 → Tp
    ((104, 105, 103, 104), (-1, 0)),       # H = Tp → 未成交，續抱（Hd 未到）
    ((105, 105, 103, 104), (-1, 0)),       # O = Tp、H = Tp → 未成交
    ((101, 104, 100, 103), (-1, 0)),       # 未觸及
])
def test_11c_take_profit_limit(bar, expect):
    o = hold([bar], 100, tp=105, stop=90, st=CLOSE, Hd=5)
    assert (o["cls"][0], o["exit_px"][0]) == expect


# ---------------------------------------------------------------- 12
def test_12_volume_rule_through_reclassified_to_V():
    bar = [(605, 606, 590, 600)]                     # 盤中穿價
    kw = dict(tp=650, stop=560, st=CLOSE, Hd=5, subs=(1000, 0, 0, 0), lp=(600,), E=1, fill_p=0.5)
    ok = entry(bar, 610, vol=100_000, V=Vpath(1, {0: [0.9]}), **kw)       # 1000 × 100 = 100,000 ≤ 量 → 穿價成交
    assert (ok["sub_px"][0, 0], ok["sub_kind"][0, 0]) == (60000, u.K_THROUGH)
    no = entry(bar, 610, vol=99_999, V=Vpath(1, {0: [0.9]}), **kw)        # 超過 1% → 改用 V，0.9 ≥ 0.5 → 不成交
    assert (no["sub_px"][0, 0], no["sub_kind"][0, 0]) == (0, u.K_TOUCHED_NOFILL)
    yes = entry(bar, 610, vol=99_999, V=Vpath(1, {0: [0.1]}), **kw)
    assert (yes["sub_px"][0, 0], yes["sub_kind"][0, 0]) == (60000, u.K_VOL_V)
    zero = entry(bar, 610, vol=0, V=Vpath(1, {0: [0.9]}), **kw)           # 量 0 → 任何委託都視為超過 1%
    assert zero["sub_px"][0, 0] == 0


# ---------------------------------------------------------------- 13
def test_13_fill_day_rules():
    kw = dict(prev=610, tp=620, Hd=5, subs=(1000, 0, 0, 0), lp=(600,), E=1)
    # a. 成交當日停利不計；次日停利（O 615 ≤ 620、H 625 ≥ 621）
    a = entry([(605, 625, 599, 615), (615, 625, 610, 620)], stop=560, st=CLOSE, **{k: v for k, v in kw.items() if k != "prev"}, prev=610)
    assert (a["cls"][0], a["exit_day"][0], a["exit_px"][0]) == (TP, 1, 62000)
    # b. 盤中停損照算：max(589, 585, 549) = 589
    b = entry([(605, 606, 585, 588)], stop=590, st=INTRA, **{k: v for k, v in kw.items() if k != "prev"}, prev=610)
    assert (b["cls"][0], b["exit_day"][0], b["exit_px"][0]) == (SL, 0, 58900)
    # c. 收盤確認照算：d0 C 588 < 590 → 待執行；d1 prev 588 DN 530 → max(585, 580, 530) = 585
    c = entry([(605, 606, 588, 588), (586, 590, 580, 585)], stop=590, st=CLOSE, **{k: v for k, v in kw.items() if k != "prev"}, prev=610)
    assert (c["cls"][0], c["exit_day"][0], c["exit_px"][0]) == (CL, 1, 58500)
    # d. Hd 從次日起算：d0 成交、Hd=2 → 第 Hd 日 = d2
    d = entry([(605, 606, 599, 600)] + [(600,) * 4] * 2, prev=610, tp=650, stop=560, st=CLOSE, Hd=2,
              subs=(1000, 0, 0, 0), lp=(600,), E=1)
    assert (d["cls"][0], d["exit_day"][0], d["exit_px"][0]) == (EXP, 2, 60000)


# ---------------------------------------------------------------- 14
def test_14_two_batches_open_vs_after_first_fill():
    days = [(605, 606, 588, 600), (600, 601, 589, 595)]
    kw = dict(tp=650, stop=560, st=CLOSE, Hd=5, subs=(1000, 0, 1000, 0), lp=(600, 590), E=3)
    op = entry(days, 610, after=0, **kw)
    assert op["sub_day"][0].tolist() == [0, -1, 0, -1] and op["shares"][0] == 2000
    af = entry(days, 610, after=1, **kw)
    assert af["sub_day"][0].tolist() == [0, -1, 1, -1]          # 第二批不得用 d0 的 OHLC
    act = [(e["day"], e["sub"]) for e in events(af["trace"][0], "ORDER_ACTIVE")]
    assert act == [(0, 0), (1, 2)]


# ---------------------------------------------------------------- 15
def test_15_q_star_cases():
    r = u.q_star(0.3, 100.0, [(0.7, -20.0)])                     # q* = 20 / 120
    assert r["q_star"] == pytest.approx(1 / 6, abs=1e-15) and r["margin_pp"] == pytest.approx((0.3 - 1 / 6) * 100)
    assert u.q_star(0.3, 100.0, [(0.7, 10.0)])["q_star"] is None          # q* = −10/90 < 0
    assert "< 0" in u.q_star(0.3, 100.0, [(0.7, 10.0)])["reason"]
    assert "> 1" in u.q_star(0.3, -5.0, [(0.7, -10.0)])["reason"]          # q* = 10/5 = 2
    assert "分母為 0" in u.q_star(0.3, -10.0, [(0.7, -10.0)])["reason"]
    assert "機率為 0" in u.q_star(1.0, 50.0, [])["reason"]


# ---------------------------------------------------------------- 16
def test_16_es_holding_entry_cond_strategy_two_valuations():
    h = mk_out([EXP, SL, SL], [100, 95, 90], [1000] * 3)
    r = u.evaluate_version(h, COST, 0, B=COST.sell_net(10000, 1000))
    # V − B：0、94580 − 99558 = −4978、89602 − 99558 = −9956；N=3、k=1
    assert r["ev"] == -4978 and r["es"] == -9956
    e = mk_out([NOF, NOF, TP, UNF], [0, 0, 620, 540], [0, 0, 1000, 1000], cons_px=[0, 0, 0, 486],
               sub_px=[[0] * 4, [0] * 4, [60000, 0, 0, 0], [60000, 0, 0, 0]])
    r = u.evaluate_version(e, COST, 1, subs=[1000, 0, 0, 0])
    # 買進 600855；TP 617257 → 16402；UNF 主要 537611 → −63244；保守 483850 → −117005
    assert r["ev_cond"] == -23421 and r["ev_strategy"] == -11710.5
    assert r["es_cond"] == -63244 and r["es_strategy"] == -63244
    assert r["ev_cond_conservative_unfinished"] == -50301.5
    assert r["es_cond_conservative_unfinished"] == -117005 and r["es_strategy_conservative_unfinished"] == -117005


# ---------------------------------------------------------------- 17
def test_17_test_A_matches_hand_calc():
    r = u.test_A({TP: 0.3, SL: 0.5, EXP: 0.2}, {TP: 100.0, SL: -50.0, EXP: -10.0})
    # TP 0.2；SL 0.5 + 0.1×5/7；EXP 0.2 + 0.1×2/7 → 20 − 28.571428… − 2.285714… = −10.857142…
    assert r["ev"] == pytest.approx(-10.857142857142857, abs=1e-12) and r["moved_pp"] == pytest.approx(10)
    r2 = u.test_A({TP: 0.05, SL: 0.95}, {TP: 100.0, SL: -10.0})      # 不足 10 → 全移：EV = −10
    assert r2["ev"] == pytest.approx(-10.0) and r2["moved_pp"] == pytest.approx(5)


# ---------------------------------------------------------------- 19（缺日偵測；其餘見整合測試）
def test_19a_stock_missing_day_detected_and_cross_gap_excluded():
    import datetime as dt
    ds = [dt.date(2026, 10, i) for i in (1, 2, 5, 6)]
    idx = [(x, Dm(1), Dm(1), Dm(1), Dm(1), 1, None) for x in ds]
    rows = [(x, Dm(100), Dm(100), Dm(100), Dm(100), 10, None) for x in ds if x != ds[2]]
    s, log = u.build_samples(rows, idx, "after_close", dt.date(2026, 10, 7), None, {})
    assert log["stock_missing_days"] == ["2026-10-05"] and [x["date"] for x in s] == [ds[1]]


def test_19b_window_plan_by_usable_days():
    assert u.plan_windows(60) == ([20, 40, 60], [])
    assert u.plan_windows(59) == ([20, 40], [60])
    with pytest.raises(u.EngineRefusal):
        u.plan_windows(19)


# ---------------------------------------------------------------- 20
@pytest.mark.parametrize("bar,px", [((590, 592, 580, 585), 58000), ((590, 592, 570, 575), 57900), ((575, 578, 570, 576), 57400)])
def test_20_triggered_sell_price_floor(bar, px):
    o = hold([bar], 600, tp=650, stop=580, st=INTRA, Hd=5)
    assert (o["cls"][0], o["exit_px"][0]) == (SL, px)


# ---------------------------------------------------------------- 22
def test_22_ohlc_repair_and_clip():
    # 比值 O 1.05、H 1.02、L 0.99、C 0.97 → 取整後 H = max(102, 105, 97) = 105、L = min(99, 105, 97) = 97
    R = np.array([[1.05, 1.02, 0.99, 0.97], [1.2, 1.25, 0.5, 0.85]])
    o = u.run_core(R, np.full(2, 10**9), np.array([[0, 1]]), None, mode=0, start_prev_c=10000, hold_shares=1000,
                   tp_c=20000, stop_c=100, stop_type=CLOSE, Hd=5, trace_n=1)
    px = prices_of(o["trace"][0])
    assert px[0][:4] == (Dm(105), Dm(105), Dm(97), Dm(97))
    # d1 prev 97：DN 87.3、UP 106.5；O 116.4→106.5、H→106.5、L 48.5→87.3、C 82.45→87.3
    assert px[1] == (Dm("106.5"), Dm("106.5"), Dm("87.3"), Dm("87.3"), Dm("87.3"), Dm("106.5"))
    assert o["clip_n"][0] == 4


# ---------------------------------------------------------------- 23
def test_23_lock_requires_all_four_and_blocks_tp():
    o = hold([(90, 90.1, 90, 90.1), (90.1,) * 4], 100, tp=110, stop=80, st=CLOSE, Hd=2)
    assert o["lock_n"][0] == 0 and not events(o["trace"][0], "LOCK")
    # O ＝ L ＝ C ＝ DN 但 H > DN（盤中曾打開）→ 不是 LOCK；第 Hd 日照常以 max(89.9, 90, 90) = 90 期滿賣出
    o = hold([(90, 92, 90, 90)], 100, tp=110, stop=80, st=CLOSE, Hd=1)
    assert o["lock_n"][0] == 0 and (o["cls"][0], o["exit_day"][0], o["exit_px"][0], o["delay"][0]) == (EXP, 0, 9000, 0)
    # 第 Hd 日 LOCK 於 109（> 停利 105）：停利不成交，期滿轉待執行，次日 max(109.5, 109.5, 98.1) = 109.5
    days = [(99, 110, 99, 110), (104, 121, 104, 121), (109,) * 4, (110, 111, 109.5, 110)]
    o = entry(days, 100, tp=105, stop=80, st=CLOSE, Hd=2, subs=(1000, 0, 1000, 0), lp=(100, 104.5), E=2, after=1)
    assert o["sub_px"][0].tolist() == [9900, 0, 10400, 0]
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0], o["delay"][0]) == (EXP, 3, 10950, 1)
    assert [e["reason"] for e in events(o["trace"][0], "PENDING")] == ["expiry"]


# ---------------------------------------------------------------- 24
def test_24_split_fees_each_with_minimum():
    assert u.split_shares(1450) == (1000, 450)
    assert COST.order(5000, 1000, "sell") == (Dm(50000), Dm(71), Dm(150))
    assert COST.order(5000, 450, "sell") == (Dm(22500), Dm(32), Dm(67))
    assert COST.sell_net(5000, 1450) == Dm(72180)
    c = u.Costs({**COSTS, "discount": "0.28"})
    assert c.order(5000, 1000, "sell")[1] == Dm(20)   # 19.95 → 19 → 最低 20
    assert c.order(5000, 450, "sell")[1] == Dm(8)     # 8.9775 → 8（> 最低 1）


# ---------------------------------------------------------------- 25
def test_25_two_batch_combinations_values():
    days = [(605, 606, 588, 600), (615, 625, 610, 620)]       # 量 0 → 全部用 V；d1 停利 620
    V = np.full((4, 2, 4), 0.99)
    V[0, 0, 0], V[0, 0, 2] = 0.1, 0.1     # 兩批
    V[1, 0, 0] = 0.1                       # 只有第一批
    V[2, 0, 2] = 0.1                       # 只有第二批
    from helpers import ratios_from_path
    o = u.run_core(ratios_from_path(610, days), np.zeros(2), np.tile(np.arange(2), (4, 1)), V, mode=1,
                   start_prev_c=61000, tp_c=62000, stop_c=56000, stop_type=CLOSE, Hd=5, sub_shares=(1000, 0, 1000, 0),
                   lp_c=(60000, 59000), E=1, fill_p=0.5)
    assert o["shares"].tolist() == [2000, 1000, 1000, 0] and o["cls"].tolist() == [TP, TP, TP, NOF]
    r = u.evaluate_version(o, COST, 1, subs=[1000, 0, 1000, 0])
    # 2000@620 = 1234513；買 600855、590840 → 42818；16402；26417；0
    assert [int(x) / 10**6 for x in o["_V_main_u"]] == [42818, 16402, 26417, 0]
    assert r["fill_combinations"] == {"full|full": 0.25, "full|none": 0.25, "none|full": 0.25, "none|none": 0.25}
    # 部位已出場後，未成交子委託不得成交：d0 盤中停損出場，第二批 d1 穿價也不成交
    o2 = entry([(605, 606, 585, 588), (600, 601, 580, 590)], 610, tp=650, stop=590, st=INTRA, Hd=5,
               subs=(1000, 0, 1000, 0), lp=(600, 580), E=3)
    assert o2["sub_px"][0].tolist() == [60000, 0, 0, 0] and o2["exit_day"][0] == 0
    assert [(e["sub"], e["reason"]) for e in events(o2["trace"][0], "ORDER_CANCEL")] == [(2, "position_pending_or_exited")]


# ---------------------------------------------------------------- 26
def test_26_unified_deadline():
    days = [(600, 600, 599, 600), (600,) * 4, (600, 600, 589, 600), (600,) * 4, (600,) * 4, (600,) * 4]
    o = entry(days, 600, tp=650, stop=560, st=CLOSE, Hd=5, subs=(1000, 0, 1000, 0), lp=(600, 590), E=5)
    assert o["sub_day"][0].tolist() == [0, -1, 2, -1]
    assert (o["cls"][0], o["exit_day"][0], o["shares"][0], o["exit_px"][0]) == (EXP, 5, 2000, 60000)


# ---------------------------------------------------------------- 27
def test_27_order_valid_E_days():
    kw = dict(tp=650, stop=560, st=CLOSE, Hd=5, subs=(1000, 0, 0, 0), lp=(600,), E=2)
    late = entry([(605,) * 4, (605,) * 4, (605, 605, 599, 600)], 605, **kw)
    assert (late["cls"][0], late["sub_px"][0, 0], late["exit_day"][0]) == (NOF, 0, -1)
    assert events(late["trace"][0], "ORDER_EXPIRE")[0]["day"] == 1
    ok = entry([(605,) * 4, (605, 605, 599, 600)], 605, **kw)
    assert ok["sub_day"][0, 0] == 1


# ---------------------------------------------------------------- 28
def test_28_cash_dividend_eligibility():
    # holding：d1 除息 5 元，前一日收盤持有 → 5 × 1000 = 5000 元
    h = hold([(605,) * 4, (600,) * 4], 605, tp=700, stop=500, st=CLOSE, Hd=2, ex_days=(1,), ex_div_u=(5_000_000,))
    assert h["div"][0] == 5_000 * 10**6
    # entry：除息日當天才成交買進 → 不取得
    e1 = entry([(605, 606, 599, 600)], 610, tp=650, stop=560, st=CLOSE, Hd=3, subs=(1000, 0, 0, 0), lp=(600,), E=1,
               ex_days=(0,), ex_div_u=(5_000_000,))
    assert e1["shares"][0] == 1000 and e1["div"][0] == 0
    # entry：d0 成交，d1 除息（前收 605 → R 600、DN 540），d1 盤中停損賣出 → 仍取得
    e2 = entry([(605, 606, 599, 605), (580, 590, 575, 585)], 610, tp=650, stop=590, st=INTRA, Hd=3,
               subs=(1000, 0, 0, 0), lp=(600,), E=1, ex_days=(1,), ex_div_u=(5_000_000,))
    assert (e2["cls"][0], e2["exit_day"][0], e2["exit_px"][0]) == (SL, 1, 57900)   # base O 580 → max(579, 575, 540)
    assert e2["div"][0] == 5_000 * 10**6


# ---------------------------------------------------------------- 29 / 45
CAL = ["2026-11-%02d" % d for d in (20, 23, 24, 25, 26, 27, 30)]


@pytest.mark.parametrize("typ,msg", [("stock_dividend", "股票股利"), ("capital_reduction", "減資")])
def test_29_45_stock_dividend_or_capital_reduction_in_horizon_refused(typ, msg):
    import datetime as dt
    ev = [{"type": typ, "ex_date": "2026-11-24", "suspension": None}]
    with pytest.raises(u.EngineRefusal, match=msg):
        u.map_future_events(ev, CAL, dt.date(2026, 11, 19), 5)
    # 期間外（第 6 個交易日之後）不影響
    assert u.map_future_events([{"type": typ, "ex_date": "2026-11-30"}], CAL, dt.date(2026, 11, 19), 5) == ([], [], [])


# ---------------------------------------------------------------- 30
def test_30_fill_interval_low_high_with_V():
    P = 10
    from helpers import ratios_from_path
    R = ratios_from_path(610, [(605, 606, 600, 602)])
    V = np.full((P, 1, 4), 0.99)
    V[:, 0, 0] = np.arange(P) / 10 + 0.05                 # 0.05, 0.15, …, 0.95
    res = {}
    for fp in (0.3, 0.7):
        o = u.run_core(R, np.full(1, 10**9), np.zeros((P, 1)), V, mode=1, start_prev_c=61000, tp_c=65000,
                       stop_c=56000, stop_type=CLOSE, Hd=5, sub_shares=(1000, 0, 0, 0), lp_c=(60000, 0), E=1, fill_p=fp)
        res[fp] = int((o["shares"] > 0).sum())
    assert res == {0.3: 3, 0.7: 7}


# ---------------------------------------------------------------- 31
def test_31_lock_day_close_confirm_creates_pending():
    # prev 611 → DN ceil(549.9) = 550；LOCK 550 < 觸發 580 → 待執行；d1 prev 550 DN 495 → max(559, 555, 495)
    o = hold([(550,) * 4, (560, 565, 555, 560)], 611, tp=650, stop=580, st=CLOSE, Hd=5)
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0]) == (CL, 1, 55900)
    assert [(e["day"], e["reason"]) for e in events(o["trace"][0], "PENDING")] == [(0, "close_confirm_exit")]


# ---------------------------------------------------------------- 32
def test_32_two_batches_same_day_touch_use_own_columns():
    o = entry([(605, 606, 600, 602)], 610, tp=650, stop=560, st=CLOSE, Hd=5, subs=(1000, 0, 1000, 0), lp=(600, 600),
              E=1, V=Vpath(1, {0: [0.1], 2: [0.9]}))
    assert o["sub_px"][0].tolist() == [60000, 0, 0, 0] and o["shares"][0] == 1000


# ---------------------------------------------------------------- 33 / 42
def test_33_42_gap_open_but_volume_exceeded_uses_V_and_fills_at_Lp():
    kw = dict(tp=650, stop=560, st=CLOSE, Hd=5, subs=(1000, 0, 0, 0), lp=(600,), E=1, vol=50_000)
    no = entry([(590, 595, 580, 592)], 610, V=Vpath(1, {0: [0.7]}), **kw)
    assert no["sub_px"][0, 0] == 0
    yes = entry([(590, 595, 580, 592)], 610, V=Vpath(1, {0: [0.2]}), **kw)
    assert (yes["sub_px"][0, 0], yes["sub_kind"][0, 0]) == (60000, u.K_VOL_V)   # 600，不是 590
    assert COST.buy_cost(60000, 1000) == Dm(600855)                              # 手續費依 600：855


# ---------------------------------------------------------------- 34
def test_34_tick_ladder():
    td = {"10": "9.99", "50": "49.95", "100": "99.9", "500": "499.5", "1000": "999"}
    for p, w in td.items():
        assert u.tick_down_dec(p) == Dm(w) and u.tick_down_i(u.d2c(p)) == u.d2c(w)
    tu = {"9.99": "10", "49.95": "50", "99.9": "100", "499.5": "500", "999": "1000", "1000": "1005",
          "10": "10.05", "50": "50.1", "100": "100.5", "500": "501"}
    for p, w in tu.items():
        assert u.tick_up_dec(p) == Dm(w) and u.tick_up_i(u.d2c(p)) == u.d2c(w)
    ties = {"9.995": "10", "10.025": "10.05", "49.975": "50", "99.95": "100", "100.25": "100.5",
            "499.75": "500", "500.5": "501", "999.5": "1000", "1002.5": "1005", "1002.4": "1000", "5.555": "5.56"}
    for x, w in ties.items():
        r = u.round_legal_dec(x)
        assert isinstance(r, Dm) and r == Dm(w), x
        assert u.round_legal_f(float(Dm(x) * 100)) == u.d2c(w), x
    for x in ("10", "50", "100", "500", "1000"):
        assert u.is_legal_dec(x)
    for x in ("10.01", "50.05", "100.1", "500.5", "1001"):
        assert not u.is_legal_dec(x)


# ---------------------------------------------------------------- 35
def test_35_stress_B_price():
    kw = dict(tp=650, stop=580, st=INTRA, Hd=5, stressB=1, amp_c=2000.0)
    a = hold([(590, 592, 570, 575)], 600, **kw)          # 原 579 → 579 − 20 = 559（可低於 L 570）
    assert a["exit_px"][0] == 55900
    b = hold([(590, 592, 570, 575)], 627, **kw)          # DN = ceil(564.3) = 565 → 565
    assert b["exit_px"][0] == 56500


# ---------------------------------------------------------------- 36
def test_36_batch2_delayed_activation_window():
    base = [(610,) * 4] * 4 + [(610, 610, 599, 605)] + [(605,) * 4] * 5
    kw = dict(tp=650, stop=560, st=CLOSE, subs=(1000, 0, 1000, 0), lp=(600, 590), E=5, after=1)
    o = entry(base + [(605, 605, 589, 600)], 610, Hd=20, **kw)                     # 第 11 日（d10）才穿價
    tr = o["trace"][0]
    assert o["sub_day"][0].tolist() == [4, -1, -1, -1]
    assert [e["day"] for e in events(tr, "ORDER_ACTIVE") if e["sub"] == 2] == [5]
    assert [e["day"] for e in events(tr, "ORDER_EXPIRE") if e["sub"] == 2] == [9]
    o9 = entry(base[:9] + [(605, 605, 589, 600)], 610, Hd=20, **kw)               # d9（第 10 日）穿價 → 成交
    assert o9["sub_day"][0, 2] == 9
    # 統一期限較早：Hd=3 → 最後可成交日 4 + 3 − 1 = d6；d7 穿價不成交
    o3 = entry(base[:7] + [(605, 605, 589, 600)], 610, Hd=3, **kw)
    assert o3["sub_px"][0, 2] == 0
    assert [(e["day"], e["reason"]) for e in events(o3["trace"][0], "ORDER_CANCEL")] == [(6, "unified_deadline")]


# ---------------------------------------------------------------- 37
def test_37_no_new_fill_from_Hd_day():
    days = [(605, 606, 599, 600), (600,) * 4, (600,) * 4, (600, 600, 590, 600)]
    o = entry(days, 610, tp=650, stop=560, st=CLOSE, Hd=3, subs=(1000, 0, 1000, 0), lp=(600, 590), E=10,
              V=Vpath(4, {2: [0.1] * 4}))
    assert o["sub_px"][0, 2] == 0 and (o["cls"][0], o["exit_day"][0], o["shares"][0]) == (EXP, 3, 1000)
    assert [(e["day"], e["sub"], e["reason"]) for e in events(o["trace"][0], "ORDER_CANCEL")] == [(2, 2, "unified_deadline")]


# ---------------------------------------------------------------- 38
def test_38_holding_ev_is_mean_V_minus_B_without_cost_basis():
    from helpers import ratios_from_path
    R = np.vstack([ratios_from_path(100, [(100, 106, 100, 105)]), ratios_from_path(100, [(100, 100, 94, 95)])])
    o = u.run_core(R, np.full(2, 10**9), np.array([[0], [1]]), None, mode=0, start_prev_c=10000, hold_shares=1000,
                   tp_c=10500, stop_c=9500, stop_type=INTRA, Hd=1)
    assert o["exit_px"].tolist() == [10500, 9490]
    r = u.evaluate_version(o, COST, 0, B=COST.sell_net(10000, 1000))
    assert r["ev"] == -49.5      # (104536 − 99558 + 94481 − 99558) / 2；不含任何買進成本


# ---------------------------------------------------------------- 39
def test_39_whole_and_odd_buy_independent():
    o = entry([(605, 606, 600, 602)], 610, tp=650, stop=560, st=CLOSE, Hd=5, subs=(1000, 450, 0, 0), lp=(600,), E=1,
              V=Vpath(1, {0: [0.1], 1: [0.9]}))
    assert o["shares"][0] == 1000 and o["sub_px"][0].tolist() == [60000, 0, 0, 0]
    o["cls"][0], o["exit_px"][0] = TP, 62000      # 只為取得組合統計
    r = u.evaluate_version(o, COST, 1, subs=[1000, 450, 0, 0])
    assert r["fill_combinations"] == {"partial|none_defined": 1.0}


# ---------------------------------------------------------------- 40
def test_40_sell_whole_and_odd_same_price_separate_fees():
    o = hold([(100, 106, 100, 105)], 100, tp=105, stop=90, st=CLOSE, Hd=5, shares=1450)
    assert (o["cls"][0], o["exit_px"][0]) == (TP, 10500)
    u.evaluate_version(o, COST, 0, B=Dm(0))
    assert net(o) == 151578        # 105000 − 149 − 315 ＋ 47250 − 67 − 141


# ---------------------------------------------------------------- 41
def test_41_ex_dividend_day_limits_from_adjusted_ref():
    o = hold([(610,) * 4, (605, 610, 600, 605)], 600, tp=700, stop=500, st=CLOSE, Hd=5, ex_days=(1,), ex_div_u=(5_000_000,))
    ex = events(o["trace"][0], "EX_DIVIDEND_DAY")[0]
    assert ex["R"] == Dm(605) and prices_of(o["trace"][0])[1][4:] == (Dm(545), Dm(665))


# ---------------------------------------------------------------- 43（plan；完整 run 見整合測試）
def test_43_window_plan():
    assert u.plan_windows(45) == ([20, 40], [60])
    assert u.plan_windows(25) == ([20], [40, 60])
    with pytest.raises(u.EngineRefusal, match="不足 20 日"):
        u.plan_windows(15)


# ---------------------------------------------------------------- 44
def test_44_V_column_map_fixed():
    from helpers import ratios_from_path
    R = ratios_from_path(610, [(605, 606, 600, 602)])
    P = 8
    V = np.random.default_rng(5).random((P, 1, 4))
    V[:, 0, 3] = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    got = []
    for subs in ((1000, 0, 0, 300), (1000, 450, 0, 300), (1000, 450, 2000, 300)):
        o = u.run_core(R, np.full(1, 10**9), np.zeros((P, 1)), V, mode=1, start_prev_c=61000, tp_c=65000, stop_c=56000,
                       stop_type=CLOSE, Hd=5, sub_shares=subs, lp_c=(60000, 60000), E=1, fill_p=0.5)
        got.append((o["sub_px"][:, 3] > 0).tolist())
    assert got[0] == got[1] == got[2] == [True, False, True, False, True, False, True, False]


# ---------------------------------------------------------------- 46
def test_46_fee_rounding_modes():
    fl = u.Costs(COSTS)
    hu = u.Costs({**COSTS, "fee_rounding": "ROUND_HALF_UP", "tax_rounding": "ROUND_HALF_UP"})
    assert fl.order(2105, 500, "sell") == (Dm(10525), Dm(14), Dm(31))    # 14.998125 / 31.575
    assert hu.order(2105, 500, "sell") == (Dm(10525), Dm(15), Dm(32))
    assert fl.order(2105, 1000, "sell") == (Dm(21050), Dm(29), Dm(63))   # 29.99625 / 63.15
    assert hu.order(2105, 1000, "sell") == (Dm(21050), Dm(30), Dm(63))
    assert fl.sell_net(2105, 1500) == Dm(31438) and hu.sell_net(2105, 1500) == Dm(31435)   # 分別取整，不合併
    with pytest.raises(ValueError):
        u.Costs({**COSTS, "fee_rounding": "ROUND_UP"})


# ---------------------------------------------------------------- 48
def test_48_sample_stats():
    w = [{"ohlc": (Dm(101), Dm(103), Dm(99), Dm(102)), "prev_ref": Dm(100), "ratios": [1.01, 1.03, 0.99, 1.02]},
         {"ohlc": (Dm(98), Dm(100), Dm(97), Dm(99)), "prev_ref": Dm(100), "ratios": [0.98, 1.0, 0.97, 0.99]},
         {"ohlc": (Dm(90), Dm(90), Dm(90), Dm(90)), "prev_ref": Dm(100), "ratios": [0.9] * 4}]
    s = u.sample_stats(w)
    assert s["avg_amplitude"] == pytest.approx(0.07 / 3) and s["adverse_gap_days"] == 2
    assert s["adverse_gap_avg"] == pytest.approx(0.06) and s["one_price_limit_down_days"] == 1
    assert s["mu"] == pytest.approx((math.log(1.02) + math.log(0.99) + math.log(0.9)) / 3)


# ---------------------------------------------------------------- 49
def test_49_batch2_fill_day_blocks_tp_then_next_day_tp_both_exit():
    days = [(605, 606, 599, 600), (600, 625, 589, 622), (618, 625, 615, 620)]
    o = entry(days, 610, tp=620, stop=560, st=CLOSE, Hd=5, subs=(1000, 0, 1000, 0), lp=(600, 590), E=3)
    assert o["sub_day"][0].tolist() == [0, -1, 1, -1]
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0], o["shares"][0]) == (TP, 2, 62000, 2000)
    u.evaluate_version(o, COST, 1, subs=[1000, 0, 1000, 0])
    assert net(o) == 42818


# ---------------------------------------------------------------- 50
def test_50_batch1_stops_before_batch2_activation():
    o = entry([(605, 606, 585, 590), (600, 601, 570, 590)], 610, tp=650, stop=590, st=INTRA, Hd=5,
              subs=(1000, 0, 1000, 0), lp=(600, 580), E=3, after=1)
    assert o["sub_px"][0].tolist() == [60000, 0, 0, 0] and (o["cls"][0], o["exit_px"][0]) == (SL, 58900)
    assert o["sub_kind"][0, 2] == u.K_NEVER_ACTIVE
    u.evaluate_version(o, COST, 1, subs=[1000, 0, 1000, 0])
    assert net(o) == -14461   # 589000 − 839 − 1767 − 600855


# ---------------------------------------------------------------- 51 / 57（解讀 A：計畫只有一種 stop_type，見報告）
@pytest.mark.parametrize("st,cls", [(INTRA, SL), (CLOSE, CL)])
def test_51_lock_on_Hd_with_stop_single_pending_single_exit(st, cls):
    o = hold([(90,) * 4, (85, 88, 84, 86)], 100, tp=110, stop=95, st=st, Hd=1)
    tr = o["trace"][0]
    assert len(events(tr, "PENDING")) == 1 and len(events(tr, "EXIT")) == 1
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0]) == (cls, 1, 8490)
    u.evaluate_version(o, COST, 0, B=Dm(0))
    assert net(o) == 84526   # 只計費一次


@pytest.mark.parametrize("st,cls,day,px", [(INTRA, SL, 0, 58900), (CLOSE, CL, 1, 58500)])
def test_57_fill_day_intraday_and_close_conditions_single_exit(st, cls, day, px):
    # 成交日 L 585 ≤ 590 且 C 588 < 590：兩種條件都成立 → 只一筆出場
    o = entry([(605, 606, 585, 588), (586, 590, 580, 585)], 610, tp=650, stop=590, st=st, Hd=5,
              subs=(1000, 0, 0, 0), lp=(600,), E=1)
    tr = o["trace"][0]
    assert len(events(tr, "EXIT")) == 1 and len(events(tr, "PENDING")) == (1 if st == CLOSE else 0)
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0]) == (cls, day, px)


# ---------------------------------------------------------------- 52
def test_52_partial_partial_combination():
    days = [(605, 606, 600, 600), (600, 601, 590, 600), (600,) * 4, (600,) * 4]
    V = Vpath(4, {0: [0.1] * 4, 1: [0.9] * 4, 2: [0.9] * 4, 3: [0.9, 0.1, 0.9, 0.9]})
    o = entry(days, 610, tp=650, stop=560, st=CLOSE, Hd=3, subs=(1000, 450, 1000, 300), lp=(600, 590), E=2, V=V, vol=0)
    assert o["sub_px"][0].tolist() == [60000, 0, 0, 59000] and o["shares"][0] == 1300
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0]) == (EXP, 3, 60000)
    r = u.evaluate_version(o, COST, 1, subs=[1000, 450, 1000, 300])
    assert r["fill_combinations"] == {"partial|partial": 1.0}
    assert net(o) == -1558    # 597345 + 179204 − 600855 − 177252；加權成本 (600000+177000)/1300


# ---------------------------------------------------------------- 53
def test_53_pending_on_Hd_locked_to_cap():
    days = [(100,) * 4, (90,) * 4] + [(x,) * 4 for x in LOCK_CHAIN_FROM_90]
    o = hold(days, 100, tp=110, stop=80, st=CLOSE, Hd=2)
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0], o["cons_px"][0]) == (UNF, 11, 3155, 2840)
    r = u.evaluate_version(o, COST, 0, B=COST.sell_net(10000, 1000))
    assert r["unfinished_paths"] == 1 and r["ev"] == -68146 and r["ev_conservative_unfinished"] == -71283
    assert events(o["trace"][0], "UNFINISHED")[0]["reason"] == "expiry"


# ---------------------------------------------------------------- 54
def test_54_batch2_fills_on_Hd_minus_1_b1_odd_cancelled():
    days = [(605, 606, 600, 600), (600,) * 4, (600, 625, 590, 600), (600,) * 4]
    V = Vpath(4, {0: [0.1] * 4, 1: [0.9] * 4, 2: [0.1] * 4})
    o = entry(days, 610, tp=620, stop=560, st=CLOSE, Hd=3, subs=(1000, 450, 1000, 0), lp=(600, 590), E=10, V=V, vol=0)
    assert o["sub_day"][0].tolist() == [0, -1, 2, -1]
    tr = o["trace"][0]
    assert [(e["day"], e["sub"], e["reason"]) for e in events(tr, "ORDER_CANCEL")] == [(2, 1, "unified_deadline")]
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0], o["shares"][0]) == (EXP, 3, 60000, 2000)
    u.evaluate_version(o, COST, 1, subs=[1000, 450, 1000, 0])
    assert net(o) == 2995     # 1194690 − 600855 − 590840


# ---------------------------------------------------------------- 55 / 56
def test_55_fill_day_intraday_stop_sells_whole_position():
    o = entry([(605, 606, 580, 600)], 610, tp=650, stop=590, st=INTRA, Hd=5, subs=(1000, 450, 0, 0), lp=(600,), E=1)
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0], o["shares"][0]) == (SL, 0, 58900, 1450)


def test_56_fill_day_close_confirm_uses_next_day_prices_only():
    o = entry([(605, 606, 560, 570), (575, 590, 572, 580)], 610, tp=650, stop=590, st=CLOSE, Hd=5,
              subs=(1000, 0, 0, 0), lp=(600,), E=1)
    # 成交日 L 560 不得使用；d1 prev 570 DN 513 → max(574, 572, 513) = 574
    assert (o["cls"][0], o["exit_day"][0], o["exit_px"][0]) == (CL, 1, 57400)


# ---------------------------------------------------------------- 58
def test_58_dividend_R_per_path():
    from helpers import ratios_from_path
    R = np.vstack([ratios_from_path(600, [(610,) * 4]), ratios_from_path(600, [(590,) * 4]), np.ones((1, 4))])
    o = u.run_core(R, np.full(3, 10**9), np.array([[0, 2], [1, 2]]), None, mode=0, start_prev_c=60000, hold_shares=1000,
                   tp_c=70000, stop_c=40000, stop_type=CLOSE, Hd=5, ex_days=(1,), ex_div_u=(5_000_000,), trace_n=2)
    a, b = prices_of(o["trace"][0])[1], prices_of(o["trace"][1])[1]
    assert a == (Dm(605),) * 4 + (Dm(545), Dm(665))       # R 605：ceil(544.5)=545、floor(665.5)=665
    assert b == (Dm(585),) * 4 + (Dm(527), Dm(643))       # R 585：ceil(526.5)=527、floor(643.5)=643


# ---------------------------------------------------------------- 59
def test_59_dividend_counted_once():
    o = hold([(605,) * 4, (605,) * 4], 610, tp=700, stop=500, st=CLOSE, Hd=2, ex_days=(0,), ex_div_u=(5_000_000,))
    assert (o["exit_day"][0], o["exit_px"][0], o["div"][0]) == (1, 60500, 5_000 * 10**6)
    r = u.evaluate_version(o, COST, 0, B=COST.sell_net(61000, 1000))
    assert net(o) == 607323 and r["ev"] == 22        # 605000 − 862 − 1815 + 5000；B = 610000 − 869 − 1830


# ---------------------------------------------------------------- 60
def test_60_es_k_and_ties():
    M = 10**6
    assert u.es_mean(np.arange(101, dtype=np.int64) * M) == 2.5            # k=6：0..5
    assert u.es_mean(np.array([-3, -3, -3, 1, 2, 3, 4], np.int64) * M) == -3   # k=1
    assert u.es_mean(np.array([7], np.int64) * M) == 7
    assert u.es_mean(np.array([-5] * 3 + [0] * 98, np.int64) * M) == -2.5  # k=6：(−15 + 0×3)/6
    assert u.es_mean(np.array([], np.int64)) is None


# ---------------------------------------------------------------- 61
def test_61_entry_probability_denominators():
    cls = [NOF] * 4 + [TP] * 3 + [SL] * 3
    px = [0] * 4 + [620] * 3 + [589] * 3
    sub = [[0] * 4] * 4 + [[60000, 0, 0, 0]] * 6
    r = u.evaluate_version(mk_out(cls, px, [0] * 4 + [1000] * 6, sub_px=sub), COST, 1, subs=[1000, 0, 0, 0])
    tp = r["classes"]["take_profit"]
    assert (tp["p_uncond"], tp["p_cond"], r["p_any_fill"]) == (0.3, 0.5, 0.6)
    assert r["classes"]["intraday_stop"]["p_cond"] == 0.5 and r["classes"]["no_fill"]["p_uncond"] == 0.4
    assert r["prob_sum"] == pytest.approx(1.0)


# ---------------------------------------------------------------- 62
def test_62_entry_unfinished_values_exclude_unfilled_subs():
    chain = [79.4, 71.5, 64.4, 58.0, 52.2, 47.0, 42.3, 38.1, 34.3, 30.9]   # 自 88.2 起逐日跌停（人工）
    days = [(100, 100, 98, 98), (88.2,) * 4] + [(x,) * 4 for x in chain]
    o = entry(days, 100, tp=110, stop=80, st=CLOSE, Hd=1, subs=(1000, 0, 1000, 0), lp=(99, 95), E=1)
    assert o["sub_px"][0].tolist() == [9900, 0, 0, 0]
    assert (o["cls"][0], o["exit_px"][0], o["cons_px"][0]) == (UNF, 3090, 2785)
    r = u.evaluate_version(o, COST, 1, subs=[1000, 0, 1000, 0])
    assert r["ev_cond"] == -68377 and r["ev_cond_conservative_unfinished"] == -71413


# ---------------------------------------------------------------- 63（build_samples；整合見另檔）
def test_63_sample_cutoff():
    import datetime as dt
    t = dt.date(2026, 11, 19)
    ds = [dt.date(2026, 11, 17), dt.date(2026, 11, 18), t]
    rows = [(d, Dm(100), Dm(101), Dm(99), Dm(c), 10, None) for d, c in zip(ds, (100, 100, 100.5))]
    idx = [(d, Dm(1), Dm(1), Dm(1), Dm(1), 1, None) for d in ds]
    s, log = u.build_samples(rows, idx, "intraday", t, None, {})
    assert log["excluded_today"] == "2026-11-19" and [x["date"] for x in s] == [ds[1]]
    s, log = u.build_samples(rows, idx, "after_close", t, 10050, {})
    assert log["after_close_today_included"] and [x["date"] for x in s] == ds[1:]
    with pytest.raises(u.EngineRefusal, match="官方收盤價"):
        u.build_samples(rows, idx, "after_close", t, 10100, {})


# ---------------------------------------------------------------- 65
def _raw(rows):
    return {"data": {"data": rows}}


GOOD = {"date": "2026-11-18", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10, "turnover": 1000}


@pytest.mark.parametrize("bad,msg", [
    ([GOOD, dict(GOOD)], "日期重複"),
    ([{**GOOD, "open": 0}], "非正數"),
    ([{**GOOD, "low": -1}], "非正數"),
    ([{**GOOD, "high": 99.5}], "不合理"),
    ([{**GOOD, "volume": -1}], "成交量"),
    ([{k: v for k, v in GOOD.items() if k != "volume"}], "volume 缺失"),
    ([{**GOOD, "date": "2026/11/18"}], "日期格式"),
])
def test_65_raw_candle_checks(bad, msg):
    with pytest.raises(u.EngineRefusal, match=msg):
        u.check_candles(_raw(bad), "t")
    assert len(u.check_candles(_raw([GOOD]), "t")) == 1


# ---------------------------------------------------------------- 66
def test_66_batch1_never_fills_batch2_never_activates():
    o = entry([(610,) * 4, (610,) * 4, (580, 590, 575, 585)], 610, tp=650, stop=560, st=CLOSE, Hd=5,
              subs=(1000, 0, 1000, 0), lp=(600, 590), E=2, after=1)
    assert o["cls"][0] == NOF and o["sub_kind"][0, 2] == u.K_NEVER_ACTIVE and o["shares"][0] == 0
    assert [(e["sub"], e["reason"]) for e in events(o["trace"][0], "ORDER_CANCEL")] == [(2, "batch1_never_filled")]


# ---------------------------------------------------------------- 67（旗標；整合見另檔）
def test_67_unverified_flag_and_unfilled_subs_free():
    assert COST.flags() == ["費用取整尚未核實，淨期望值含未確認的成本假設"]
    assert u.Costs({**COSTS, "fee_rounding_verified": True, "tax_rounding_verified": True}).flags() == []
    # 只有第一批成交的路徑：Vᵢ 只扣第一批買進成本（第二批不計任何費用）
    r = u.evaluate_version(mk_out([TP], [620], [1000], sub_px=[[60000, 0, 0, 0]]), COST, 1, subs=[1000, 450, 1000, 300])
    assert r["ev_cond"] == 16402
