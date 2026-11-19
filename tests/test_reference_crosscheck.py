"""獨立數值驗證：依規格 §3.2、§4.2、§4.4～§4.8 另寫一份「純 Python ＋ Decimal」的參考模擬
（不呼叫引擎的任何價格或狀態函數，只用 Python 標準庫），在隨機路徑上與 numba 引擎逐路徑比對。
涵蓋：holding（兩種停損方式）、entry 單批整股（含 V、成交量前提、E 期滿）。"""
from decimal import Decimal as Dm, ROUND_FLOOR, ROUND_CEILING
import numpy as np
import pytest

from helpers import u

BANDS = [(Dm(10), Dm("0.01")), (Dm(50), Dm("0.05")), (Dm(100), Dm("0.1")), (Dm(500), Dm("0.5")), (Dm(1000), Dm(1))]


def tick(x):
    for ub, t in BANDS:
        if x < ub:
            return t
    return Dm(5)


def fl(x):
    t = tick(x); return (x / t).to_integral_value(ROUND_FLOOR) * t


def ce(x):
    t = tick(x); return (x / t).to_integral_value(ROUND_CEILING) * t


def tdown(p): return fl(p - Dm("0.01"))
def tup(p): return ce(p + Dm("0.01"))


def rlegal(x):
    lo, hi = fl(x), ce(x)
    return lo if lo == hi or (x - lo) < (hi - x) else hi


def bar(ratio_row, ref):
    dn, up = ce(ref * Dm("0.9")), fl(ref * Dm("1.1"))
    o, h, l, c = (min(max(rlegal(Dm(float(r)) * ref), dn), up) for r in ratio_row)
    h, l = max(h, o, c), min(l, o, c)
    return o, h, l, c, dn, (o == h == l == c == dn)


def ref_path(R, vols, idx_row, V_row, *, entry, start, tp, stop, intraday, Hd, lp=None, E=0, p=0.0, sh=1000):
    prev, pend, delay = start, None, 0
    held = not entry
    fill_day = -1 if not entry else None
    for d, j in enumerate(idx_row):
        o, h, l, c, dn, lock = bar(R[j], prev)
        if pend is not None:
            if lock:
                delay += 1
            else:
                return pend, d, max(tdown(o), l, dn), delay, fill_day
            prev = c
            continue
        fill_today = False
        if entry and not held:
            if d < E:
                volok = sh * 100 <= vols[j]
                fill = None
                if o < lp:
                    fill = o if volok else (lp if V_row[d] < p else None)
                elif l <= tdown(lp):
                    fill = lp if volok or V_row[d] < p else None
                elif l == lp:
                    fill = lp if V_row[d] < p else None
                if fill is not None:
                    held, fill_today, fill_day = True, True, d
            if not held:
                if d >= E - 1:
                    return "nofill", -1, None, 0, None
                prev = c
                continue
        hh = d - fill_day
        tp_ok, tp_px = False, None
        if not fill_today and not lock:
            if o > tp:
                tp_ok, tp_px = True, o
            elif h >= tup(tp):
                tp_ok, tp_px = True, tp
        if intraday and l <= stop:
            if lock:
                pend = "sl"; delay += 1
            else:
                return "sl", d, max(tdown(o if o < stop else stop), l, dn), delay, fill_day
        elif tp_ok:
            return "tp", d, tp_px, delay, fill_day
        elif (not intraday) and c < stop:
            pend = "close"
        if pend is None and hh == Hd:
            if lock:
                pend = "exp"; delay += 1
            else:
                return "exp", d, max(tdown(c), l, dn), delay, fill_day
        if pend is not None and hh >= Hd + 10:
            return "unf", d, dn, delay, fill_day
        prev = c
    return "open", -1, None, delay, fill_day


NAME = {u.CLS_TP: "tp", u.CLS_SL: "sl", u.CLS_CLOSE: "close", u.CLS_EXP: "exp", u.CLS_UNF: "unf", u.CLS_NOFILL: "nofill"}


def random_market(seed, n=40):
    rng = np.random.default_rng(seed)
    R = np.exp(rng.normal(0, 0.035, size=(n, 4)))
    R[: n // 8] = 0.85                    # 跌停鎖死日（會被夾到 DN）
    R[n // 8: n // 6] = [1.0, 1.12, 0.86, 1.05]   # 漲跌停都觸及
    vols = rng.integers(0, 300_000, size=n)
    return R, vols


@pytest.mark.parametrize("start,intraday", [(Dm("612"), True), (Dm("612"), False), (Dm("48.7"), True), (Dm("101.5"), False)])
def test_holding_matches_reference(start, intraday):
    R, vols = random_market(int(start * 10) + intraday)
    P, Hd = 1500, 5
    idx = np.random.default_rng(7).integers(0, len(R), size=(P, Hd + 11))
    tp, stop = fl(start * Dm("1.08")), ce(start * Dm("0.94"))
    o = u.run_core(R, vols, idx, None, mode=0, start_prev_c=int(start * 100), hold_shares=1000, tp_c=int(tp * 100),
                   stop_c=int(stop * 100), stop_type=0 if intraday else 1, Hd=Hd)
    for i in range(P):
        k, d, px, delay, _ = ref_path(R, vols, idx[i], None, entry=False, start=start, tp=tp, stop=stop, intraday=intraday, Hd=Hd)
        assert NAME[int(o["cls"][i])] == k and o["exit_day"][i] == d and o["delay"][i] == delay, i
        assert Dm(int(o["exit_px"][i])) / 100 == px, i


@pytest.mark.parametrize("intraday,p", [(True, 0.3), (False, 0.7)])
def test_entry_single_batch_matches_reference(intraday, p):
    start = Dm("612")
    R, vols = random_market(99 + intraday)
    P, Hd, E = 1500, 4, 3
    rng = np.random.default_rng(8)
    idx = rng.integers(0, len(R), size=(P, E + Hd + 11))
    V = rng.random((P, E + Hd + 11, 4))
    lp, tp, stop = Dm(600), Dm(640), Dm(580)
    o = u.run_core(R, vols, idx, V, mode=1, start_prev_c=61200, tp_c=64000, stop_c=58000, stop_type=0 if intraday else 1,
                   Hd=Hd, sub_shares=(1000, 0, 0, 0), lp_c=(60000, 0), E=E, fill_p=p)
    nf = 0
    for i in range(P):
        k, d, px, delay, ff = ref_path(R, vols, idx[i], V[i, :, 0], entry=True, start=start, tp=tp, stop=stop,
                                       intraday=intraday, Hd=Hd, lp=lp, E=E, p=p)
        assert NAME[int(o["cls"][i])] == k, (i, k)
        nf += k == "nofill"
        if k != "nofill":
            assert o["first_fill"][i] == ff and o["exit_day"][i] == d and o["delay"][i] == delay, i
            assert Dm(int(o["exit_px"][i])) / 100 == px, i
    assert 0 < nf < P      # 兩種結果都有出現，比對才有意義
