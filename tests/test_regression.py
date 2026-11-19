"""GPT 程式審查 R1～R3 的回歸測試，以及整數／Decimal 等價、可重現性、升降單位邊界、事件追蹤整合測試。"""
import datetime as dt
from decimal import Decimal as Dm, ROUND_FLOOR, ROUND_CEILING

import numpy as np
import pytest

from helpers import u, one_path, C, COSTS, check_path_built, events, prices_of, weekdays

COST = u.Costs(COSTS)
TODAY = dt.date(2026, 11, 19)
CAL = [d.isoformat() for d in weekdays(dt.date(2026, 11, 20), 30)]     # 11/20 起 30 個平日（測試用日曆）
CASH = lambda date, amt="5": {"type": "cash_dividend", "ex_date": date, "cash_per_share": amt}


# ======================================================== R1 交易日曆涵蓋
class TestR1Calendar:
    D = 14

    def test_no_future_event_needs_no_calendar(self):
        assert u.map_future_events([], None, TODAY, self.D) == ([], [], [])
        assert u.map_future_events([CASH("2026-11-19")], None, TODAY, self.D) == ([], [], [])   # 今天或以前 → 不屬持有期間

    @pytest.mark.parametrize("cal,msg", [(None, "未提供官方交易日曆"), ([], "未提供官方交易日曆"),
                                         (CAL[:13], "不足以涵蓋"),
                                         ([CAL[1], CAL[0]] + CAL[2:], "嚴格遞增"),
                                         ([CAL[0], CAL[0]] + CAL[1:], "嚴格遞增"),
                                         (["2026-11-19"] + CAL, "第一天")])
    def test_bad_calendar_refused(self, cal, msg):
        with pytest.raises(u.EngineRefusal, match=msg):
            u.map_future_events([CASH(CAL[40 // 4])], cal, TODAY, self.D)

    def test_event_beyond_calendar_end_not_silently_ignored(self):
        # 舊版：cal[min(D,len)-1] 只看到日曆尾端，日曆太短時期間內事件會被當成期間外
        with pytest.raises(u.EngineRefusal, match="不足以涵蓋"):
            u.map_future_events([CASH(CAL[12])], CAL[:5], TODAY, self.D)

    def test_horizon_boundaries(self):
        assert u.map_future_events([CASH(CAL[13])], CAL, TODAY, self.D)[:2] == ([13], [5_000_000])   # 第 D 個交易日：納入
        assert u.map_future_events([CASH(CAL[14])], CAL, TODAY, self.D)[:2] == ([], [])               # 第 D+1 個：期間外
        assert u.map_future_events([CASH(CAL[0])], CAL, TODAY, self.D)[:2] == ([0], [5_000_000])

    def test_non_trading_ex_date_and_bad_amounts(self):
        with pytest.raises(u.EngineRefusal, match="不是交易日"):
            u.map_future_events([CASH("2026-11-22")], CAL, TODAY, self.D)          # 週日
        for bad, msg in ((None, "不完整"), ("-1", "非負數"), ("NaN", "非負數"), ("abc", "無法解析")):
            with pytest.raises(u.EngineRefusal, match=msg):
                u.map_future_events([CASH(CAL[2], bad)], CAL, TODAY, self.D)
        with pytest.raises(u.EngineRefusal, match="無法辨識"):
            u.map_future_events([{"type": "rights_issue", "ex_date": CAL[2]}], CAL, TODAY, self.D)

    def test_quantization_flag(self):
        d, a, f = u.map_future_events([CASH(CAL[2], "1.2345678")], CAL, TODAY, self.D)
        assert a == [1_234_568] and f and "0.000001" in f[0]


# ======================================================== R2 同日多筆現金股利
def test_R2_same_date_dividends_are_summed():
    ev = [CASH(CAL[1], "3"), CASH(CAL[1], "2")]
    d, a, _ = u.map_future_events(ev, CAL, TODAY, 14)
    assert (d, a) == ([1, 1], [3_000_000, 2_000_000])
    days = [(605,) * 4, (600,) * 4]
    kw = dict(mode=0, hold_shares=1000, tp_c=C(700), stop_c=C(500), stop_type=1, Hd=2)
    two = one_path(days, 605, ex_days=(1, 1), ex_div_u=(3_000_000, 2_000_000), **kw)
    one = one_path(days, 605, ex_days=(1,), ex_div_u=(5_000_000,), **kw)
    check_path_built(two, days)
    ex = events(two["trace"][0], "EX_DIVIDEND_DAY")[0]
    assert ex["R"] == Dm(600) and ex["div"] == Dm(5)           # R = 605 − (3 + 2)
    assert two["div"][0] == one["div"][0] == 5_000 * 10**6
    assert prices_of(two["trace"][0]) == prices_of(one["trace"][0])


def test_R2_dividend_accrual_exact_integer_and_non_cent_R():
    # 1450 股 × 0.123456 元 = 179.0112 元（精確）；R = 100 − 0.123456 = 99.876544（非整分）
    days = [(99.9, 99.9, 99.9, 99.9)]
    o = one_path(days, 100, mode=0, hold_shares=1450, tp_c=C(120), stop_c=C(50), stop_type=1, Hd=1,
                 ex_days=(0,), ex_div_u=(123_456,))
    assert o["div"][0] == 1450 * 123_456
    dn, up = u.limits_dec(Dm("99.876544"))
    assert prices_of(o["trace"][0])[0][4:] == (dn, up) == (Dm("89.9"), Dm("109.5"))
    u.evaluate_version(o, COST, 0, B=Dm(0))
    # 期滿賣出 max(tick_down(99.9)=99.8, L 99.9, DN) = 99.9：whole 99900−142−299、odd 44955−64−134
    assert o["exit_px"][0] == 9990
    assert Dm(int(o["_V_main_u"][0])) / 10**6 == Dm(99459) + Dm(44757) + Dm("179.0112")


# ======================================================== R3 滑價說明
def test_R3_tax_note_wording():
    n = u.TAX_NOTE
    assert n.startswith("本引擎損益已扣除交易手續費、證交稅及模型設定之滑價，但未扣除個人股利所得稅及二代健保補充保費。")
    for frag in ("§4.2", "max(tick_down(基準價), 當日最低價, 跌停價)", "至多 1 個升降單位", "限價單（進場、停利）不扣滑價",
                 "測試 B", "不包含在一般模擬路徑的損益中"):
        assert frag in n


# ======================================================== 整數分 ↔ Decimal 等價
def test_int_vs_decimal_ladder_exhaustive_to_2500():
    c = np.arange(1, 250_001)
    for x in c.tolist():
        d = Dm(x) / 100
        assert u.floor_legal_i(x) == int(u.floor_legal_dec(d) * 100)
        assert u.ceil_legal_i(x) == int(u.ceil_legal_dec(d) * 100)
        assert u.tick_down_i(x) == int(u.tick_down_dec(d) * 100) if x > 1 else True
        assert u.tick_up_i(x) == int(u.tick_up_dec(d) * 100)


def test_limits_micro_vs_decimal():
    for x in range(1, 200_001):                      # 0.01～2000.00 每一分
        dn, up = u.limits_dec(Dm(x) / 100)
        assert u.limits_micro(x * 10_000) == (int(dn * 100), int(up * 100)), x
    rng = np.random.default_rng(1)
    for m in rng.integers(10_000, 3_000_000_000, size=20_000).tolist():   # 1e-6 元參考價（除息後）
        dn, up = u.limits_dec(Dm(m) / 10**6)
        assert u.limits_micro(m) == (int(dn * 100), int(up * 100)), m


def test_round_legal_float_vs_decimal():
    # 等距點：每個區間取多個 k·t + t/2（以分表示可精確表示），必須取高
    for lo, hi, t in ((1, 1000, 1), (1000, 5000, 5), (5000, 10000, 10), (10000, 50000, 50), (50000, 100000, 100), (100000, 300000, 500)):
        for k in range(lo // t, hi // t, max(1, (hi - lo) // t // 200)):
            x = k * t + t / 2
            want = u.round_legal_dec(Dm(x) / 100)
            assert u.round_legal_f(float(x)) == int(want * 100), x
    rng = np.random.default_rng(2)
    for x in rng.uniform(1, 300_000, size=50_000).tolist():
        want = u.round_legal_dec(Dm(x) / 100)          # Dm(x) 為 float 的精確二進位值
        assert u.round_legal_f(x) == int(want * 100), x


def test_tick_boundaries_and_band_of_result():
    # 漲跌停的升降單位依「計算結果」所在區間
    assert u.limits_dec("11.11") == (Dm("10"), Dm("12.2"))     # 9.999 → ceil 0.01 → 10.00
    assert u.limits_dec("9.1") == (Dm("8.19"), Dm("10"))       # 10.01 → 落在 10～50 區間 → floor 0.05 → 10.00
    assert u.limits_dec("455") == (Dm("409.5"), Dm("500"))     # 500.5 → 落在 500～1000 → floor 1 → 500
    assert u.limits_dec("909.1") == (Dm("819"), Dm("1000"))    # 1000.01 → ≥1000 → floor 5 → 1000
    assert u.limits_dec("1111") == (Dm("1000"), Dm("1220"))    # 999.9 → ceil 1 → 1000


# ======================================================== 可重現性
def test_core_deterministic_repeatability():
    rng = np.random.default_rng(4)
    R = np.exp(rng.normal(0, 0.04, size=(40, 4)))
    U, V = u.make_uniforms(99, 2000, 16, True)
    U2, V2 = u.make_uniforms(99, 2000, 16, True)
    assert np.array_equal(U, U2) and np.array_equal(V, V2)
    idx = u.window_index(U, 40)
    kw = dict(mode=1, start_prev_c=60000, tp_c=66000, stop_c=55000, stop_type=0, Hd=3, sub_shares=(1000, 450, 0, 300),
              lp_c=(59000, 58000), E=2, batch2_after=1, fill_p=0.5)
    a = u.run_core(R, np.full(40, 10**7), idx, V, **kw)
    b = u.run_core(R, np.full(40, 10**7), idx, V, **kw)
    for k in a:
        if k != "trace":
            assert np.array_equal(a[k], b[k]), k


# ======================================================== 事件追蹤整合（完整事件序列）
def test_event_trace_full_sequence_two_batches():
    # 規格 §7 測試 49 的路徑：完整事件序列（價格事件除外）
    days = [(605, 606, 599, 600), (600, 625, 589, 622), (618, 625, 615, 620)]
    o = one_path(days, 610, mode=1, tp_c=C(620), stop_c=C(560), stop_type=1, Hd=5, sub_shares=(1000, 0, 1000, 0),
                 lp_c=(C(600), C(590)), E=3, batch2_after=0, fill_p=0.5, V=np.full((1, 3, 4), 0.99))
    seq = [(e["day"], e["event"], e.get("sub"), str(e.get("price", "")), e.get("class", ""))
           for e in o["trace"][0] if e["event"] != "PRICES"]
    assert seq == [(0, "ORDER_ACTIVE", 0, "", ""), (0, "ORDER_ACTIVE", 2, "", ""),
                   (0, "FILL", 0, "600", ""), (0, "POSITION_OPEN", None, "", ""),
                   (1, "FILL", 2, "590", ""),
                   (2, "EXIT", None, "620", "take_profit")]
