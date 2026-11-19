"""uep_sim.py — UEP 模擬引擎（依 uep_sim_spec_v5.md，2026-11-19 凍結版實作）

用途：UEP 第四步的持股評估（holding）與新倉評估（entry）模擬。
界線：本程式通過測試只代表「在規格與測試條件下計算一致」，不代表模擬機率準確，
      也不代表策略有正期望值。驗證只能靠實際交易紀錄扣成本後與 0050 比較。

實作說明（交審查者確認）：
- 價格一律以「整數分（cent）」表示，所有合法價位運算在整數上完成，與 §3.1 的 Decimal 版本
  （tick_down_dec 等公開函數）逐分比對測試一致。§3.1 要求「以 Decimal 實作、不得用二進位浮點比較價位」，
  整數分同樣是精確表示；引擎內部為了速度（numba 編譯）使用整數版本。
- 抽樣比值 × 前收得到的「原始價格」本質上是實數（漂移調整含 exp），以 float64 計算後
  經 round_legal 轉成精確的整數分；等距判斷容差 1e-7 分。
- 費用以 decimal.Decimal 計算後依 costs 指定的方式取整到元。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import sys
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP, ROUND_CEILING

import numpy as np

try:
    from numba import njit, prange
    _NUMBA = True
except Exception:  # numba 不可用時退回純 Python（慢，但結果相同）
    _NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        if a and callable(a[0]):
            return a[0]
        return deco

    prange = range

VERSION = "1.0.1"
SPEC = "uep_sim_spec_v5.md (2026-11-19 frozen)"

# ---------------------------------------------------------------------------
# 1. 價格階梯 — Decimal 參考版本（§3.1）
# ---------------------------------------------------------------------------
_BANDS_DEC = [(Decimal("10"), Decimal("0.01")), (Decimal("50"), Decimal("0.05")),
              (Decimal("100"), Decimal("0.1")), (Decimal("500"), Decimal("0.5")),
              (Decimal("1000"), Decimal("1"))]
_TOP_TICK_DEC = Decimal("5")


def tick_size_dec(x: Decimal) -> Decimal:
    """x 所在價格區間的升降單位。"""
    for upper, t in _BANDS_DEC:
        if x < upper:
            return t
    return _TOP_TICK_DEC


def floor_legal_dec(x) -> Decimal:
    x = Decimal(str(x))
    t = tick_size_dec(x)
    return (x / t).to_integral_value(rounding=ROUND_FLOOR) * t


def ceil_legal_dec(x) -> Decimal:
    x = Decimal(str(x))
    t = tick_size_dec(x)
    return (x / t).to_integral_value(rounding=ROUND_CEILING) * t


def is_legal_dec(x) -> bool:
    x = Decimal(str(x))
    return x > 0 and floor_legal_dec(x) == x


def tick_down_dec(p) -> Decimal:
    """嚴格小於 p 的最大合法價位。"""
    p = Decimal(str(p))
    return floor_legal_dec(p - Decimal("0.01"))


def tick_up_dec(p) -> Decimal:
    """嚴格大於 p 的最小合法價位。"""
    p = Decimal(str(p))
    return ceil_legal_dec(p + Decimal("0.01"))


def round_legal_dec(x) -> Decimal:
    """最接近 x 的合法價位；等距時取較高者。"""
    x = Decimal(str(x))
    lo = floor_legal_dec(x)
    if lo == x:
        return lo
    hi = ceil_legal_dec(x)
    return lo if (x - lo) < (hi - x) else hi


def limits_dec(ref) -> tuple:
    """(跌停, 漲停)：UP = floor_legal(ref×1.1)、DN = ceil_legal(ref×0.9)。"""
    ref = Decimal(str(ref))
    return ceil_legal_dec(ref * Decimal("0.9")), floor_legal_dec(ref * Decimal("1.1"))


# ---------------------------------------------------------------------------
# 2. 價格階梯 — 整數分版本（引擎內部使用）
# ---------------------------------------------------------------------------
@njit(cache=False)
def tick_c(x):
    if x < 1000:
        return 1
    if x < 5000:
        return 5
    if x < 10000:
        return 10
    if x < 50000:
        return 50
    if x < 100000:
        return 100
    return 500


@njit(cache=False)
def floor_legal_i(p):
    t = tick_c(p)
    return (p // t) * t


@njit(cache=False)
def ceil_legal_i(p):
    t = tick_c(p)
    return -((-p) // t) * t


@njit(cache=False)
def tick_down_i(p):
    return floor_legal_i(p - 1)


@njit(cache=False)
def tick_up_i(p):
    return ceil_legal_i(p + 1)


@njit(cache=False)
def round_legal_f(x):
    """實數分 x → 最近的合法價位（整數分）；等距取高。"""
    t = tick_c(x)
    q = x / t
    qf = math.floor(q)
    if q - qf > 1.0 - 1e-9:
        qf += 1.0
    lo = qf * t
    if abs(x - lo) < 1e-7:
        return np.int64(lo + 0.5)
    hi = lo + t
    if (x - lo) < (hi - x) - 1e-7:
        return np.int64(lo + 0.5)
    return np.int64(hi + 0.5)


@njit(cache=False)
def _tick_micro(v_num_over10_lt):
    return 0  # placeholder (not used)


@njit(cache=False)
def limits_micro(refm):
    """refm：參考價（單位 1e-6 元）。回傳 (DN, UP)，單位分。"""
    # 價格區間門檻（1e-6 元）與升降單位（1e-6 元）
    # <10:1e4, <50:5e4, <100:1e5, <500:5e5, <1000:1e6, >=1000:5e6
    # UP
    vu = refm * 11  # = 10 × (ref×1.1)
    if vu < 10 * 10_000_000:
        tu = 10_000
    elif vu < 10 * 50_000_000:
        tu = 50_000
    elif vu < 10 * 100_000_000:
        tu = 100_000
    elif vu < 10 * 500_000_000:
        tu = 500_000
    elif vu < 10 * 1_000_000_000:
        tu = 1_000_000
    else:
        tu = 5_000_000
    up_m = (vu // (10 * tu)) * tu
    vd = refm * 9
    if vd < 10 * 10_000_000:
        td = 10_000
    elif vd < 10 * 50_000_000:
        td = 50_000
    elif vd < 10 * 100_000_000:
        td = 100_000
    elif vd < 10 * 500_000_000:
        td = 500_000
    elif vd < 10 * 1_000_000_000:
        td = 1_000_000
    else:
        td = 5_000_000
    dn_m = -((-vd) // (10 * td)) * td
    return dn_m // 10_000, up_m // 10_000


def c2d(c: int) -> Decimal:
    return Decimal(int(c)) / Decimal(100)


def d2c(x) -> int:
    v = Decimal(str(x)) * 100
    if v != v.to_integral_value():
        raise ValueError(f"價格 {x} 不是 0.01 的整數倍")
    return int(v)


# ---------------------------------------------------------------------------
# 3. 模擬核心（狀態機）
# ---------------------------------------------------------------------------
# 部位狀態
ST_NONE, ST_HELD, ST_PENDING, ST_EXITED, ST_UNFINISHED, ST_NOFILL = 0, 1, 2, 3, 4, 5
# 出場類別
CLS_TP, CLS_SL, CLS_CLOSE, CLS_EXP, CLS_UNF, CLS_NOFILL = 0, 1, 2, 3, 4, 5
CLS_NAMES = {0: "take_profit", 1: "intraday_stop", 2: "close_confirm_exit", 3: "expiry",
             4: "unfinished_after_extension", 5: "no_fill"}
# 子委託狀態
SUB_WAIT, SUB_ACTIVE, SUB_FILLED, SUB_CANCEL, SUB_EXPIRED, SUB_NONE = 0, 1, 2, 3, 4, 9
# 子委託最終分類
K_NA, K_GAP, K_THROUGH, K_TOUCH_V, K_VOL_V, K_TOUCHED_NOFILL, K_NEVER_TOUCHED, K_NEVER_ACTIVE = 0, 1, 2, 3, 4, 5, 6, 7
# 事件代碼
EV_ACTIVE, EV_FILL, EV_OPEN, EV_LOCK, EV_PENDING, EV_EXIT, EV_CANCEL, EV_EXPIRE, EV_DIV, EV_UNF, EV_TOUCH_FAIL, EV_EXDAY, EV_PRICES = range(1, 14)
EV_NAMES = {1: "ORDER_ACTIVE", 2: "FILL", 3: "POSITION_OPEN", 4: "LOCK", 5: "PENDING", 6: "EXIT",
            7: "ORDER_CANCEL", 8: "ORDER_EXPIRE", 9: "DIVIDEND", 10: "UNFINISHED", 11: "TOUCH_NO_FILL",
            12: "EX_DIVIDEND_DAY", 13: "PRICES"}
MAXE = 160


@njit(cache=False)
def _trace(trace, tc, p, trace_n, d, code, a, b, c):
    if p < trace_n:
        k = tc[p]
        if k < trace.shape[1]:
            trace[p, k, 0] = d
            trace[p, k, 1] = code
            trace[p, k, 2] = a
            trace[p, k, 3] = b
            trace[p, k, 4] = c
            tc[p] = k + 1


@njit(cache=False, parallel=False)
def simulate_core(ratios, vols, idx, V, mode, start_prev_c, hold_shares, tp_c, stop_c, stop_type,
                  Hd, ext_cap, sub_shares, sub_batch, lp_c, E, batch2_after, fill_p,
                  ex_days, ex_div_u, stressB, amp_c, trace_n, trace, tc,
                  o_cls, o_exit_day, o_exit_px, o_cons_px, o_delay, o_shares, o_div, o_first_fill,
                  o_sub_px, o_sub_day, o_sub_kind, o_dev_sum, o_dev_n, o_clip_n, o_lock_n):
    P = idx.shape[0]
    D = idx.shape[1]
    nex = ex_days.shape[0]
    for p in prange(P):
        state = ST_HELD if mode == 0 else ST_NONE
        shares = hold_shares if mode == 0 else 0
        first_fill = -1 if mode == 0 else -9999
        pend_reason = -1
        delay = 0
        div = 0
        sub_state = np.full(4, SUB_NONE, np.int64)
        sub_start = np.full(4, -1, np.int64)
        touched = np.zeros(4, np.int64)
        activated = np.zeros(4, np.int64)
        b2_started = 0
        for s in range(4):
            o_sub_px[p, s] = 0
            o_sub_day[p, s] = -1
            o_sub_kind[p, s] = K_NA
            if mode == 1 and sub_shares[s] > 0:
                if sub_batch[s] == 0 or batch2_after == 0:
                    sub_state[s] = SUB_ACTIVE
                    sub_start[s] = 0
                    activated[s] = 1
                    _trace(trace, tc, p, trace_n, 0, EV_ACTIVE, s, 0, 0)
                else:
                    sub_state[s] = SUB_WAIT
        prev = start_prev_c
        o_cls[p] = -1
        o_exit_day[p] = -1
        o_exit_px[p] = 0
        o_cons_px[p] = 0
        dev_sum = 0.0
        dev_n = 0
        clip_n = 0
        lock_n = 0
        for d in range(D):
            j = idx[p, d]
            refm = prev * 10_000
            exday = False
            dk = 0
            for k in range(nex):
                if ex_days[k] == d:
                    refm -= ex_div_u[k]
                    exday = True
                    dk += ex_div_u[k]
            dn, up = limits_micro(refm)
            reff = refm / 10_000.0
            px4 = np.empty(4, np.int64)
            for q in range(4):
                raw = ratios[j, q] * reff
                r = round_legal_f(raw)
                dev_sum += abs(r - raw) / raw
                dev_n += 1
                if r < dn:
                    r = dn
                    clip_n += 1
                elif r > up:
                    r = up
                    clip_n += 1
                px4[q] = r
            O = px4[0]
            H = px4[1]
            L = px4[2]
            C = px4[3]
            if H < O:
                H = O
            if H < C:
                H = C
            if L > O:
                L = O
            if L > C:
                L = C
            lock = (O == dn) and (H == dn) and (L == dn) and (C == dn)
            if lock:
                lock_n += 1
            _trace(trace, tc, p, trace_n, d, EV_PRICES, O * 1000000 + H, L * 1000000 + C, dn * 1000000 + up)
            if exday:
                _trace(trace, tc, p, trace_n, d, EV_EXDAY, refm, dk, 0)
                if (state == ST_HELD or state == ST_PENDING) and shares > 0:
                    div += shares * dk
                    _trace(trace, tc, p, trace_n, d, EV_DIV, shares, dk, 0)
            if lock and (state == ST_HELD or state == ST_PENDING):
                _trace(trace, tc, p, trace_n, d, EV_LOCK, dn, 0, 0)
            fill_today = False
            # ---- 1. 待出場 ----
            if state == ST_PENDING:
                if lock:
                    delay += 1
                else:
                    pxs = max(tick_down_i(O), L, dn)
                    if stressB == 1:
                        pxs = max(round_legal_f(pxs - amp_c), dn)
                    state = ST_EXITED
                    o_cls[p] = pend_reason
                    o_exit_day[p] = d
                    o_exit_px[p] = pxs
                    _trace(trace, tc, p, trace_n, d, EV_EXIT, pend_reason, pxs, shares)
            # ---- 2. 委託（只在未持有／持有時） ----
            if state == ST_NONE or state == ST_HELD:
                for s in range(4):
                    if sub_state[s] == SUB_WAIT and sub_start[s] >= 0 and d >= sub_start[s]:
                        sub_state[s] = SUB_ACTIVE
                        activated[s] = 1
                        _trace(trace, tc, p, trace_n, d, EV_ACTIVE, s, 0, 0)
                for s in range(4):
                    if sub_state[s] != SUB_ACTIVE:
                        continue
                    Lp = lp_c[sub_batch[s]]
                    ss = sub_shares[s]
                    volok = ss * 100 <= vols[j]
                    filled = False
                    fpx = 0
                    kind = K_NA
                    if O < Lp:
                        touched[s] = 1
                        if volok:
                            filled = True
                            fpx = O
                            kind = K_GAP
                        elif V[p, d, s] < fill_p:
                            filled = True
                            fpx = Lp
                            kind = K_VOL_V
                    elif L <= tick_down_i(Lp):
                        touched[s] = 1
                        if volok:
                            filled = True
                            fpx = Lp
                            kind = K_THROUGH
                        elif V[p, d, s] < fill_p:
                            filled = True
                            fpx = Lp
                            kind = K_VOL_V
                    elif L == Lp:
                        touched[s] = 1
                        if V[p, d, s] < fill_p:
                            filled = True
                            fpx = Lp
                            kind = K_TOUCH_V
                    if filled:
                        sub_state[s] = SUB_FILLED
                        o_sub_px[p, s] = fpx
                        o_sub_day[p, s] = d
                        o_sub_kind[p, s] = kind
                        shares += ss
                        fill_today = True
                        if first_fill < 0:
                            first_fill = d
                        _trace(trace, tc, p, trace_n, d, EV_FILL, s, fpx, kind)
                        if sub_batch[s] == 0 and batch2_after == 1 and b2_started == 0:
                            b2_started = 1
                            for s2 in range(4):
                                if sub_batch[s2] == 1 and sub_state[s2] == SUB_WAIT:
                                    sub_start[s2] = d + 1
                    elif O < Lp or L <= Lp:
                        _trace(trace, tc, p, trace_n, d, EV_TOUCH_FAIL, s, 0, 0)
                if state == ST_NONE and shares > 0:
                    state = ST_HELD
                    _trace(trace, tc, p, trace_n, d, EV_OPEN, shares, 0, 0)
            # ---- 3. 持有中的出場判斷（§4.5 第 2～5 步）----
            if state == ST_HELD:
                h = d - first_fill
                tp_ok = False
                tp_px = 0
                if (not fill_today) and (not lock):
                    if O > tp_c:
                        tp_ok = True
                        tp_px = O
                    elif H >= tick_up_i(tp_c):
                        tp_ok = True
                        tp_px = tp_c
                sl_ok = (stop_type == 0) and (L <= stop_c)
                if sl_ok:
                    if lock:
                        state = ST_PENDING
                        pend_reason = CLS_SL
                        delay += 1
                        _trace(trace, tc, p, trace_n, d, EV_PENDING, CLS_SL, 0, 0)
                    else:
                        base = O if O < stop_c else stop_c
                        pxs = max(tick_down_i(base), L, dn)
                        if stressB == 1:
                            pxs = max(round_legal_f(pxs - amp_c), dn)
                        state = ST_EXITED
                        o_cls[p] = CLS_SL
                        o_exit_day[p] = d
                        o_exit_px[p] = pxs
                        _trace(trace, tc, p, trace_n, d, EV_EXIT, CLS_SL, pxs, shares)
                elif tp_ok:
                    state = ST_EXITED
                    o_cls[p] = CLS_TP
                    o_exit_day[p] = d
                    o_exit_px[p] = tp_px
                    _trace(trace, tc, p, trace_n, d, EV_EXIT, CLS_TP, tp_px, shares)
                elif stop_type == 1 and C < stop_c:
                    state = ST_PENDING
                    pend_reason = CLS_CLOSE
                    _trace(trace, tc, p, trace_n, d, EV_PENDING, CLS_CLOSE, 0, 0)
                if state == ST_HELD and h == Hd:
                    if lock:
                        state = ST_PENDING
                        pend_reason = CLS_EXP
                        delay += 1
                        _trace(trace, tc, p, trace_n, d, EV_PENDING, CLS_EXP, 0, 0)
                    else:
                        pxs = max(tick_down_i(C), L, dn)
                        if stressB == 1:
                            pxs = max(round_legal_f(pxs - amp_c), dn)
                        state = ST_EXITED
                        o_cls[p] = CLS_EXP
                        o_exit_day[p] = d
                        o_exit_px[p] = pxs
                        _trace(trace, tc, p, trace_n, d, EV_EXIT, CLS_EXP, pxs, shares)
            # ---- 4. 委託取消／期滿 ----
            for s in range(4):
                if sub_state[s] == SUB_ACTIVE or sub_state[s] == SUB_WAIT:
                    if state == ST_PENDING or state == ST_EXITED:
                        sub_state[s] = SUB_CANCEL
                        _trace(trace, tc, p, trace_n, d, EV_CANCEL, s, 1, 0)
                        continue
                if sub_state[s] == SUB_ACTIVE:
                    if d >= sub_start[s] + E - 1:
                        sub_state[s] = SUB_EXPIRED
                        _trace(trace, tc, p, trace_n, d, EV_EXPIRE, s, 0, 0)
                    elif first_fill >= 0 and d >= first_fill + Hd - 1:
                        sub_state[s] = SUB_CANCEL
                        _trace(trace, tc, p, trace_n, d, EV_CANCEL, s, 2, 0)
                elif sub_state[s] == SUB_WAIT:
                    if sub_start[s] >= 0:
                        if first_fill >= 0 and sub_start[s] > first_fill + Hd - 1:
                            sub_state[s] = SUB_CANCEL
                            _trace(trace, tc, p, trace_n, d, EV_CANCEL, s, 2, 0)
                    else:
                        # 「第一批成交後才生效」：第一批已無任何有效子委託且未成交 → 永不啟動
                        b1_live = False
                        b1_filled = False
                        for s1 in range(4):
                            if sub_batch[s1] == 0 and sub_shares[s1] > 0:
                                if sub_state[s1] == SUB_ACTIVE:
                                    b1_live = True
                                if sub_state[s1] == SUB_FILLED:
                                    b1_filled = True
                        if (not b1_live) and (not b1_filled):
                            sub_state[s] = SUB_CANCEL
                            _trace(trace, tc, p, trace_n, d, EV_CANCEL, s, 3, 0)
            # ---- 5. 結束條件 ----
            if state == ST_EXITED:
                break
            if state == ST_NONE:
                any_live = False
                for s in range(4):
                    if sub_state[s] == SUB_ACTIVE or sub_state[s] == SUB_WAIT:
                        any_live = True
                if not any_live:
                    state = ST_NOFILL
                    o_cls[p] = CLS_NOFILL
                    break
            if state == ST_PENDING:
                h = d - first_fill
                if h >= Hd + ext_cap:
                    state = ST_UNFINISHED
                    o_cls[p] = CLS_UNF
                    o_exit_day[p] = d
                    o_exit_px[p] = dn
                    cdn, cup = limits_micro(dn * 10_000)
                    o_cons_px[p] = cdn
                    _trace(trace, tc, p, trace_n, d, EV_UNF, dn, cdn, pend_reason)
                    break
            prev = C
        if state == ST_NONE:  # 防呆：D 用完仍未成交（不應發生）
            o_cls[p] = CLS_NOFILL
        # 子委託最終分類
        for s in range(4):
            if sub_shares[s] <= 0:
                o_sub_kind[p, s] = K_NA
            elif sub_state[s] == SUB_FILLED:
                pass
            elif activated[s] == 0:
                o_sub_kind[p, s] = K_NEVER_ACTIVE
            elif touched[s] == 1:
                o_sub_kind[p, s] = K_TOUCHED_NOFILL
            else:
                o_sub_kind[p, s] = K_NEVER_TOUCHED
        o_delay[p] = delay
        o_shares[p] = shares
        o_div[p] = div
        o_first_fill[p] = first_fill
        o_dev_sum[p] = dev_sum
        o_dev_n[p] = dev_n
        o_clip_n[p] = clip_n
        o_lock_n[p] = lock_n


def run_core(ratios, vols, idx, V, *, mode, start_prev_c, hold_shares=0, tp_c, stop_c, stop_type,
             Hd, ext_cap=10, sub_shares=(0, 0, 0, 0), lp_c=(0, 0), E=0, batch2_after=0,
             fill_p=0.0, ex_days=(), ex_div_u=(), stressB=0, amp_c=0.0, trace_n=0):
    """包裝 simulate_core：配置輸出陣列並執行。所有價格以整數分傳入。"""
    P, D = idx.shape
    ratios = np.ascontiguousarray(ratios, dtype=np.float64)
    vols = np.ascontiguousarray(vols, dtype=np.int64)
    idx = np.ascontiguousarray(idx, dtype=np.int64)
    if V is None:
        V = np.ones((1, 1, 4), dtype=np.float64)
    out = {
        "cls": np.zeros(P, np.int64), "exit_day": np.zeros(P, np.int64), "exit_px": np.zeros(P, np.int64),
        "cons_px": np.zeros(P, np.int64), "delay": np.zeros(P, np.int64), "shares": np.zeros(P, np.int64),
        "div": np.zeros(P, np.int64), "first_fill": np.zeros(P, np.int64),
        "sub_px": np.zeros((P, 4), np.int64), "sub_day": np.zeros((P, 4), np.int64),
        "sub_kind": np.zeros((P, 4), np.int64), "dev_sum": np.zeros(P, np.float64),
        "dev_n": np.zeros(P, np.int64), "clip_n": np.zeros(P, np.int64), "lock_n": np.zeros(P, np.int64),
    }
    trace = np.zeros((max(trace_n, 1), MAXE, 5), np.int64)
    tc = np.zeros(max(trace_n, 1), np.int64)
    simulate_core(ratios, vols, idx, V, int(mode), int(start_prev_c), int(hold_shares), int(tp_c), int(stop_c),
                  int(stop_type), int(Hd), int(ext_cap), np.array(sub_shares, np.int64),
                  np.array([0, 0, 1, 1], np.int64), np.array(lp_c, np.int64), int(E), int(batch2_after),
                  float(fill_p), np.array(ex_days, np.int64), np.array(ex_div_u, np.int64), int(stressB),
                  float(amp_c), int(trace_n), trace, tc,
                  out["cls"], out["exit_day"], out["exit_px"], out["cons_px"], out["delay"], out["shares"],
                  out["div"], out["first_fill"], out["sub_px"], out["sub_day"], out["sub_kind"],
                  out["dev_sum"], out["dev_n"], out["clip_n"], out["lock_n"])
    out["trace"] = [decode_trace(trace[i, :tc[i]]) for i in range(trace_n)]
    return out


def decode_trace(rows):
    ev = []
    for d, code, a, b, c in rows.tolist():
        name = EV_NAMES.get(code, str(code))
        if code == EV_PRICES:
            ev.append({"day": d, "event": name, "O": c2d(a // 1000000), "H": c2d(a % 1000000),
                       "L": c2d(b // 1000000), "C": c2d(b % 1000000), "DN": c2d(c // 1000000), "UP": c2d(c % 1000000)})
        elif code == EV_FILL:
            ev.append({"day": d, "event": name, "sub": a, "price": c2d(b), "kind": c})
        elif code == EV_EXIT:
            ev.append({"day": d, "event": name, "class": CLS_NAMES[a], "price": c2d(b), "shares": c})
        elif code == EV_PENDING:
            ev.append({"day": d, "event": name, "reason": CLS_NAMES[a]})
        elif code == EV_UNF:
            ev.append({"day": d, "event": name, "main_px": c2d(a), "cons_px": c2d(b), "reason": CLS_NAMES[c]})
        elif code == EV_LOCK:
            ev.append({"day": d, "event": name, "DN": c2d(a)})
        elif code == EV_DIV:
            ev.append({"day": d, "event": name, "shares": a, "div_per_share": Decimal(b) / Decimal(10**6), "cash": Decimal(a * b) / Decimal(10**6)})
        elif code == EV_EXDAY:
            ev.append({"day": d, "event": name, "R": Decimal(a) / Decimal(10**6), "div": Decimal(b) / Decimal(10**6)})
        elif code in (EV_ACTIVE, EV_EXPIRE, EV_TOUCH_FAIL):
            ev.append({"day": d, "event": name, "sub": a})
        elif code == EV_CANCEL:
            ev.append({"day": d, "event": name, "sub": a, "reason": {1: "position_pending_or_exited", 2: "unified_deadline", 3: "batch1_never_filled"}.get(b, b)})
        elif code == EV_OPEN:
            ev.append({"day": d, "event": name, "shares": a})
        else:
            ev.append({"day": d, "event": name, "a": a, "b": b, "c": c})
    return ev


# ---------------------------------------------------------------------------
# 4. 費用（§4.3）
# ---------------------------------------------------------------------------
_RMODE = {"ROUND_FLOOR": ROUND_FLOOR, "ROUND_HALF_UP": ROUND_HALF_UP}


class Costs:
    def __init__(self, cfg: dict):
        self.rate = Decimal(str(cfg["commission_rate"]))
        self.discount = Decimal(str(cfg["discount"]))
        self.min_whole = Decimal(str(cfg["min_fee_whole"]))
        self.min_odd = Decimal(str(cfg["min_fee_odd"]))
        self.tax = Decimal(str(cfg["tax_rate"]))
        self.fee_round = cfg["fee_rounding"]
        self.tax_round = cfg["tax_rounding"]
        self.fee_verified = bool(cfg["fee_rounding_verified"])
        self.tax_verified = bool(cfg["tax_rounding_verified"])
        if self.fee_round not in _RMODE or self.tax_round not in _RMODE:
            raise ValueError("fee_rounding / tax_rounding 必須明確指定 ROUND_FLOOR 或 ROUND_HALF_UP")
        self._cache = {}

    def _fee(self, amount: Decimal, odd: bool) -> Decimal:
        f = (amount * self.rate * self.discount).quantize(Decimal(1), rounding=_RMODE[self.fee_round])
        return max(f, self.min_odd if odd else self.min_whole)

    def _taxv(self, amount: Decimal) -> Decimal:
        return (amount * self.tax).quantize(Decimal(1), rounding=_RMODE[self.tax_round])

    def order(self, px_c: int, shares: int, side: str):
        """單一子委託：回傳 (成交金額, 手續費, 證交稅)，Decimal 元。"""
        key = (px_c, shares, side)
        r = self._cache.get(key)
        if r is None:
            amt = c2d(px_c) * shares
            odd = shares % 1000 != 0
            if shares >= 1000 and odd:
                raise ValueError("子委託必須是純整股或純零股")
            fee = self._fee(amt, odd)
            tax = self._taxv(amt) if side == "sell" else Decimal(0)
            r = (amt, fee, tax)
            self._cache[key] = r
        return r

    def sell_net(self, px_c: int, total_shares: int) -> Decimal:
        """依 §4.3 拆單（整股 1 筆＋零股 1 筆）賣出的淨收入。"""
        key = ("net", px_c, total_shares)
        r = self._cache.get(key)
        if r is None:
            k, rem = divmod(total_shares, 1000)
            r = Decimal(0)
            for sh in ((k * 1000) if k else 0, rem):
                if sh > 0:
                    amt, fee, tax = self.order(px_c, sh, "sell")
                    r += amt - fee - tax
            self._cache[key] = r
        return r

    def buy_cost(self, px_c: int, shares: int) -> Decimal:
        amt, fee, _ = self.order(px_c, shares, "buy")
        return amt + fee

    def flags(self):
        out = []
        if not (self.fee_verified and self.tax_verified):
            out.append("費用取整尚未核實，淨期望值含未確認的成本假設")
        return out

    def describe(self):
        return {"commission_rate": str(self.rate), "discount": str(self.discount),
                "min_fee_whole": str(self.min_whole), "min_fee_odd": str(self.min_odd),
                "tax_rate": str(self.tax), "fee_rounding": self.fee_round, "tax_rounding": self.tax_round,
                "fee_rounding_verified": self.fee_verified, "tax_rounding_verified": self.tax_verified}


def split_shares(total: int):
    k, r = divmod(int(total), 1000)
    return (k * 1000 if k else 0), r


# ---------------------------------------------------------------------------
# 5. 資料處理（§2）
# ---------------------------------------------------------------------------
class EngineRefusal(Exception):
    """規格要求拒絕執行的情況。"""


def _parse_date(s):
    try:
        return _dt.date.fromisoformat(str(s)[:10]) if len(str(s)) >= 10 and str(s)[4] == "-" else None
    except Exception:
        return None


def check_candles(raw: dict, label: str):
    """§2.0：原始日 K 檢查。回傳升冪排序的列表 [(date, O,H,L,C (cents), volume, turnover)]。"""
    if not isinstance(raw, dict) or "data" not in raw:
        raise EngineRefusal(f"{label}: 缺少 data 欄位")
    data = raw["data"]
    rows = data.get("data") if isinstance(data, dict) else None
    if rows is None:
        raise EngineRefusal(f"{label}: 缺少 data.data")
    out = []
    seen = set()
    for r in rows:
        for f in ("date", "open", "high", "low", "close", "volume"):
            if f not in r or r[f] is None:
                raise EngineRefusal(f"{label}: 必要欄位 {f} 缺失（{r.get('date')}）")
        d = _parse_date(r["date"])
        if d is None or len(str(r["date"])) != 10:
            raise EngineRefusal(f"{label}: 日期格式無法解析 {r['date']}")
        if d in seen:
            raise EngineRefusal(f"{label}: 日期重複 {d}")
        seen.add(d)
        o, h, l, c = (Decimal(str(r[f])) for f in ("open", "high", "low", "close"))
        if min(o, h, l, c) <= 0:
            raise EngineRefusal(f"{label}: {d} OHLC 有非正數")
        if not (l <= min(o, c) and h >= max(o, c)):
            raise EngineRefusal(f"{label}: {d} OHLC 不合理（L ≤ min(O,C) 且 H ≥ max(O,C) 不成立）")
        vol = r["volume"]
        if not isinstance(vol, (int, float)) or vol < 0:
            raise EngineRefusal(f"{label}: {d} 成交量缺失或為負數")
        out.append((d, o, h, l, c, int(vol), r.get("turnover")))
    n0 = len(out)
    out.sort(key=lambda x: x[0])
    assert len(out) == n0
    return out


def volume_unit_check(rows, label):
    """以成交值／成交量是否落在 [L, H] 判斷成交量單位為「股」。"""
    checked = 0
    for d, o, h, l, c, vol, to in rows:
        if vol > 0 and to:
            avg = Decimal(str(to)) / Decimal(vol)
            if not (l * Decimal("0.999") <= avg <= h * Decimal("1.001")):
                if l * Decimal("0.999") <= avg / 1000 <= h * Decimal("1.001"):
                    raise EngineRefusal(f"{label}: 成交量單位疑似為張，需要統一換算後再執行（{d}）")
                raise EngineRefusal(f"{label}: 成交量單位無法辨識（{d}，均價 {avg:.2f} 不在 [{l}, {h}]）")
            checked += 1
    if checked == 0:
        raise EngineRefusal(f"{label}: 無法驗證成交量單位（缺 turnover）")
    return "shares"


def build_samples(stock_rows, index_rows, run_phase, today, official_close_c, hist_refs):
    """§2.0.1、§2.1、§2.2：建立可用樣本日。
    hist_refs: {date: 參考價(Decimal)} 樣本期間的除權息／減資參考價；事件日缺參考價 → 排除。
    回傳 (samples, log)。samples: list of dict(date, ratios(4), vol, prev_ref, ohlc)。"""
    log = {"excluded_today": None, "stock_missing_days": [], "excluded_no_ref": [], "after_close_today_included": False}
    rows = list(stock_rows)
    if run_phase == "intraday":
        if rows and rows[-1][0] == today:
            log["excluded_today"] = str(today)
            rows = rows[:-1]
        rows = [r for r in rows if r[0] < today]
    elif run_phase == "after_close":
        if rows and rows[-1][0] == today:
            if d2c(rows[-1][4]) != official_close_c:
                raise EngineRefusal(f"after_close：今日日 K 收盤 {rows[-1][4]} 與官方收盤價 {c2d(official_close_c)} 不一致")
            log["after_close_today_included"] = True
    idx_dates = [r[0] for r in index_rows]
    stock_dates = {r[0] for r in rows}
    if rows:
        lo, hi = rows[0][0], rows[-1][0]
        for dd in idx_dates:
            if lo <= dd <= hi and dd not in stock_dates:
                log["stock_missing_days"].append(str(dd))
    idx_pos = {dd: i for i, dd in enumerate(idx_dates)}
    samples = []
    for i in range(1, len(rows)):
        d0, d1 = rows[i - 1][0], rows[i][0]
        if d0 not in idx_pos or d1 not in idx_pos:
            raise EngineRefusal(f"個股交易日 {d0 if d0 not in idx_pos else d1} 不在 IX0001 交易日中，無法比對")
        if idx_pos[d1] - idx_pos[d0] != 1:
            continue  # 跨個股缺日的報酬不納入
        prev_c = rows[i - 1][4]
        if d1 in hist_refs:
            ref = hist_refs[d1]
            if ref is None:
                log["excluded_no_ref"].append(str(d1))
                continue
            prev_c = Decimal(str(ref))
        d, o, h, l, c, vol, _ = rows[i]
        samples.append({"date": d1, "ratios": [float(o / prev_c), float(h / prev_c), float(l / prev_c), float(c / prev_c)],
                        "vol": vol, "prev_ref": prev_c, "ohlc": (o, h, l, c)})
    return samples, log


def plan_windows(n_usable):
    """§2.1 表：回傳 (可執行樣本窗, 無法完成的樣本窗)。"""
    if n_usable < 20:
        raise EngineRefusal(f"可用樣本日只有 {n_usable} 日（不足 20 日），拒絕執行")
    run = [w for w in (20, 40, 60) if n_usable >= w]
    skip = [w for w in (20, 40, 60) if n_usable < w]
    return run, skip


def sample_stats(win):
    amps = [float((s["ohlc"][1] - s["ohlc"][2]) / s["prev_ref"]) for s in win]
    gaps = [float((s["prev_ref"] - s["ohlc"][0]) / s["prev_ref"]) for s in win if s["ohlc"][0] < s["prev_ref"]]
    locks = 0
    for s in win:
        dn, _ = limits_dec(s["prev_ref"])
        o, h, l, c = s["ohlc"]
        if o == h == l == c == dn:
            locks += 1
    mu = sum(math.log(s["ratios"][3]) for s in win) / len(win)
    return {"n": len(win), "mu": mu, "avg_amplitude": sum(amps) / len(amps), "adverse_gap_days": len(gaps),
            "adverse_gap_avg": (sum(gaps) / len(gaps)) if gaps else 0.0, "one_price_limit_down_days": locks}


def drift_ratios(win, f, mu):
    r = np.array([s["ratios"] for s in win], dtype=np.float64)
    return np.exp(np.log(r) - (1.0 - f) * mu)


# ---------------------------------------------------------------------------
# 6. 抽樣與雜湊（§2.4）
# ---------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def make_uniforms(seed, P, D, need_V):
    U = np.random.Generator(np.random.PCG64(seed)).random((P, D), dtype=np.float64)
    V = np.random.Generator(np.random.PCG64(seed + 1)).random((P, D, 4), dtype=np.float64) if need_V else None
    return U, V


def window_index(U, n):
    return np.floor(U * n).astype(np.int32)


def dates_hash(idx, win_dates):
    arr = np.array([d.isoformat() for d in win_dates])
    joined = "\n".join(arr[idx.ravel()].tolist())
    return sha256_bytes(joined.encode("utf-8"))


# ---------------------------------------------------------------------------
# 7. 評估（§5）
# ---------------------------------------------------------------------------
def es_mean(vals):
    """§5.2：k = ceil(0.05 N)，最差 k 條的算術平均；N = 0 → None（無路徑）。"""
    v = np.asarray(vals)
    N = v.size
    if N == 0:
        return None
    k = -(-N // 20)  # ceil(0.05 N)，整數運算避免浮點誤差
    w = np.sort(v)[:k]
    if np.issubdtype(v.dtype, np.integer):  # 1e-6 元整數 → 精確平均後換算成元
        return int(w.astype(object).sum()) / k / 1_000_000
    return float(w.astype(np.float64).mean())


def q_star(p_tp, m_tp, others):
    """§5.2 損益兩平。others: list of (p, mean)。回傳 dict。"""
    p_non = sum(p for p, _ in others)
    if p_non <= 0:
        return {"q_star": None, "reason": "非停利類機率為 0，模型條件下無可行解"}
    m_non = sum(p * m for p, m in others) / p_non
    if m_tp is None:
        return {"q_star": None, "reason": "停利類無路徑，模型條件下無可行解"}
    denom = m_tp - m_non
    if denom == 0:
        return {"q_star": None, "reason": "M_停利 = M_非停利（分母為 0），模型條件下無可行解", "M_tp": m_tp, "M_non": m_non}
    q = -m_non / denom
    if q < 0:
        return {"q_star": None, "reason": f"q* = {q:.6f} < 0（非停利情境平均已為正），模型條件下無可行解", "M_tp": m_tp, "M_non": m_non}
    if q > 1:
        return {"q_star": None, "reason": f"q* = {q:.6f} > 1（停利平均不足以抵銷），模型條件下無可行解", "M_tp": m_tp, "M_non": m_non}
    return {"q_star": q, "M_tp": m_tp, "M_non": m_non, "margin_pp": (p_tp - q) * 100}


def test_A(p_by_cls, m_by_cls, tp_cls=CLS_TP):
    """§5.2 測試 A：停利機率移 10 個百分點到其他出場類別（依原比例），用各類平均重算。"""
    p_tp = p_by_cls.get(tp_cls, 0.0)
    move = min(0.10, p_tp)
    others = {c: p for c, p in p_by_cls.items() if c != tp_cls}
    tot = sum(others.values())
    newp = dict(p_by_cls)
    newp[tp_cls] = p_tp - move
    if tot > 0:
        for c, p in others.items():
            newp[c] = p + move * p / tot
    elif move > 0:
        return {"ev": None, "reason": "沒有其他出場類別可接收移轉機率"}
    ev = sum(newp[c] * (m_by_cls[c] if m_by_cls.get(c) is not None else 0.0) for c in newp)
    return {"ev": ev, "moved_pp": move * 100}


def _u(x: Decimal) -> int:
    """Decimal 元 → 整數 1e-6 元（精確；價格為整數分、費用為整數元、股利為 1e-6 元）。"""
    v = x * 1_000_000
    assert v == v.to_integral_value(), x
    return int(v)


def mean_exact(arr_u):
    """整數（1e-6 元）陣列的精確平均，回傳元（float）。"""
    n = len(arr_u)
    if n == 0:
        return None
    return int(np.asarray(arr_u, dtype=object).sum()) / n / 1_000_000


def evaluate_version(out, costs: Costs, mode, B=None, shares_hold=0, subs=None):
    """把模擬輸出轉成 §5.1 的 Vᵢ，並計算 §5.2 統計。回傳 dict（核心數值）。
    每條路徑的價值以整數（1e-6 元）精確計算；平均與 ES 以整數精確加總後才換算成元。"""
    cls = out["cls"]
    P = cls.size
    exit_px = out["exit_px"]
    cons_px = out["cons_px"]
    shares = out["shares"]
    div = out["div"]  # 1e-6 元（股數 × 每股股利 1e-6 元）
    Vm = np.zeros(P, np.int64)
    Vc = np.zeros(P, np.int64)
    buy = np.zeros(P, np.int64)
    if mode == 1:
        for s in range(4):
            ss = subs[s]
            if ss <= 0:
                continue
            fp = out["sub_px"][:, s]
            for u in np.unique(fp[fp > 0]):
                buy[fp == u] += _u(costs.buy_cost(int(u), ss))
    has = (cls != CLS_NOFILL) & (shares > 0)
    netmap = {}
    for px, sh in np.unique(np.stack([exit_px, shares], axis=1)[has], axis=0):
        netmap[(int(px), int(sh))] = _u(costs.sell_net(int(px), int(sh)))
    consmap = {}
    unf = cls == CLS_UNF
    for px, sh in np.unique(np.stack([cons_px, shares], axis=1)[unf], axis=0):
        consmap[(int(px), int(sh))] = _u(costs.sell_net(int(px), int(sh)))
    for i in np.nonzero(has)[0]:
        n = netmap[(int(exit_px[i]), int(shares[i]))]
        Vm[i] = n + div[i] - buy[i]
        Vc[i] = (consmap[(int(cons_px[i]), int(shares[i]))] if cls[i] == CLS_UNF else n) + div[i] - buy[i]
    out["_V_main_u"] = Vm
    out["_V_cons_u"] = Vc
    res = {}
    if mode == 0:
        Bu = _u(Decimal(str(B))) if not isinstance(B, Decimal) else _u(B)
        X = Vm - Bu
        Xc = Vc - Bu
        res["ev"] = mean_exact(X)
        res["ev_conservative_unfinished"] = mean_exact(Xc)
        res["es"] = es_mean(X)
        res["es_conservative_unfinished"] = es_mean(Xc)
        base_mask = np.ones(P, bool)
        vals = X
    else:
        filled = cls != CLS_NOFILL
        res["p_any_fill"] = float(filled.mean())
        res["ev_strategy"] = mean_exact(Vm)
        res["ev_strategy_conservative_unfinished"] = mean_exact(Vc)
        res["es_strategy"] = es_mean(Vm)
        res["es_strategy_conservative_unfinished"] = es_mean(Vc)
        res["ev_cond"] = mean_exact(Vm[filled])
        res["ev_cond_conservative_unfinished"] = mean_exact(Vc[filled])
        res["es_cond"] = es_mean(Vm[filled])
        res["es_cond_conservative_unfinished"] = es_mean(Vc[filled])
        base_mask = filled
        vals = Vm
    # 出場類別
    classes = [CLS_TP, CLS_SL, CLS_CLOSE, CLS_EXP, CLS_UNF] + ([CLS_NOFILL] if mode == 1 else [])
    p_unc, p_cond, m_cls, px_cls = {}, {}, {}, {}
    nb = base_mask.sum()
    for c in classes:
        m = cls == c
        p_unc[c] = float(m.mean())
        if c != CLS_NOFILL:
            p_cond[c] = float((m & base_mask).sum() / nb) if nb else 0.0
        m_cls[c] = mean_exact(vals[m])
        px_cls[c] = float(exit_px[m].mean() / 100) if m.any() and c != CLS_NOFILL else None
    res["classes"] = {CLS_NAMES[c]: {"p_uncond" if mode == 1 else "p": p_unc[c],
                                     **({"p_cond": p_cond[c]} if (mode == 1 and c != CLS_NOFILL) else {}),
                                     "mean_value": m_cls[c], "mean_exit_price": px_cls[c]} for c in classes}
    res["prob_sum"] = float(sum(p_unc.values()))
    # 出場價 10% 分位、順延天數分布
    exited = (cls != CLS_NOFILL)
    stops = np.isin(cls, [CLS_SL, CLS_CLOSE])
    res["stop_exit_price_p10"] = float(np.percentile(exit_px[stops] / 100, 10)) if stops.any() else None
    dl = out["delay"][exited]
    res["delay_distribution"] = {"0": float((dl == 0).mean()) if dl.size else None,
                                 "1": float((dl == 1).mean()) if dl.size else None,
                                 "2+": float((dl >= 2).mean()) if dl.size else None}
    # q* 與測試 A（holding：全部路徑；entry：已成交路徑的條件機率）
    pc = p_unc if mode == 0 else p_cond
    others = [(pc[c], m_cls[c]) for c in (CLS_SL, CLS_CLOSE, CLS_EXP, CLS_UNF) if pc[c] > 0]
    res["breakeven"] = q_star(pc[CLS_TP], m_cls[CLS_TP], others)
    res["test_A"] = test_A({c: pc[c] for c in (CLS_TP, CLS_SL, CLS_CLOSE, CLS_EXP, CLS_UNF)},
                           {c: m_cls[c] for c in (CLS_TP, CLS_SL, CLS_CLOSE, CLS_EXP, CLS_UNF)})
    res["win_rate"] = float((vals[base_mask] > 0).mean()) if nb else None
    res["unfinished_paths"] = int((cls == CLS_UNF).sum())
    res["price_rounding"] = {"mean_rel_dev": float(out["dev_sum"].sum() / max(out["dev_n"].sum(), 1)),
                             "clipped_prices": int(out["clip_n"].sum())}
    res["simulated_lock_days"] = int(out["lock_n"].sum())
    if mode == 1:
        names = ["batch1_whole", "batch1_odd", "batch2_whole", "batch2_odd"]
        kinds = {1: "filled_gap_open", 2: "filled_trade_through", 3: "filled_exact_touch_V", 4: "filled_volume_reclass_V",
                 5: "touched_not_filled", 6: "never_touched", 7: "never_activated"}
        sub_stats = {}
        for s in range(4):
            if subs[s] <= 0:
                continue
            k = out["sub_kind"][:, s]
            sub_stats[names[s]] = {v: float((k == kk).mean()) for kk, v in kinds.items()}
        res["sub_orders"] = sub_stats
        # 成交組合
        def bstat(b):
            ex = [s for s in (2 * b, 2 * b + 1) if subs[s] > 0]
            if not ex:
                return np.full(P, "none_defined", dtype=object)
            f = np.stack([out["sub_px"][:, s] > 0 for s in ex], axis=1)
            st = np.where(f.all(axis=1), "full", np.where(f.any(axis=1), "partial", "none"))
            return st
        b1, b2 = bstat(0), bstat(1)
        combo = np.char.add(np.char.add(b1.astype(str), "|"), b2.astype(str))
        uq, cnt = np.unique(combo, return_counts=True)
        res["fill_combinations"] = {str(u): float(c_ / P) for u, c_ in zip(uq, cnt)}
        res["es_by_combination"] = {str(u): es_mean(Vm[combo == u]) for u in uq if not str(u).startswith("none|none")}
        sh_u, sh_c = np.unique(shares[filled], return_counts=True)
        res["filled_shares_distribution"] = {int(a): float(b_ / P) for a, b_ in zip(sh_u, sh_c)}
    return res


def stress_deterministic(costs: Costs, mode, start_c, shares, B=None, lp_c=None, buy_shares=None):
    """§5.3 確定性壓力情境 A／B／C。"""
    out = {}
    dn1, _ = limits_micro(start_c * 10_000) if mode == 0 else limits_micro(lp_c * 10_000)
    dn2, _ = limits_micro(dn1 * 10_000)
    dn3, _ = limits_micro(dn2 * 10_000)
    for name, px in (("A_next_day_limit_down", dn1), ("B_two_limit_downs", dn2), ("C_two_locked_then_third_day_limit", dn3)):
        if mode == 0:
            val = float(costs.sell_net(int(px), shares) - B)
        else:
            val = float(costs.sell_net(int(px), buy_shares) - sum(costs.buy_cost(lp_c, sh) for sh in split_shares(buy_shares) if sh > 0))
        out[name] = {"exit_price": float(px) / 100, "pnl": round(val), "loss": round(-val)}
    out["note"] = ("確定性壓力情境，不給發生機率，不混入模擬分布。C 假設第三日開板且有成交，不是保證成交價。"
                   + ("entry 只評估第一批全部成交，不代表完整兩批策略的最壞情境。" if mode == 1 else ""))
    return out


# ---------------------------------------------------------------------------
# 8. 主流程 run()
# ---------------------------------------------------------------------------
TAX_NOTE = ("本引擎損益已扣除交易手續費、證交稅及模型設定之滑價，但未扣除個人股利所得稅及二代健保補充保費。"
            "（「模型設定之滑價」僅指 §4.2：觸發後執行的賣出（停損、收盤確認型出場、期滿、待執行賣出）成交價為 "
            "max(tick_down(基準價), 當日最低價, 跌停價)，即至多 1 個升降單位且不低於當日最低價；限價單（進場、停利）不扣滑價。"
            "測試 B 的振幅惡化與確定性壓力情境 A／B／C 是另外列出的壓力假設，不包含在一般模擬路徑的損益中。）")
LIMITS = [
    "真正提供資訊的只有 20～60 個樣本日；路徑數再多也不會增加歷史資訊，也不會產生樣本中不存在的風險類型（例如鎖死）。",
    "每日獨立抽樣，不含波動群聚、市場狀態轉換與事件衝擊。",
    "跌停能否成交、限價剛好觸及能否成交、同日事件先後，都只能由日 K 推定，屬模型假設。",
    "不建模部分成交；零股以整股日 K 為成交代理；停利不套用成交量前提；買進的整股與零股獨立判斷成交，不代表真實市場的兩種流動性彼此獨立。",
    "任一子委託成交當日，整體部位不執行停利（即使既有部位已持有多日），會使停利機率偏低。",
    "立即出清基準 B 以最佳委買價估算，未考慮委買深度、分價成交及市場衝擊。",
    "entry 的確定性壓力情境只評估第一批全部成交。",
    "持有期間含股票股利除權或任何減資事件時不模擬。",
    "工具資料由 Claude 轉存，雜湊只證明轉存後未被改動。",
    "intraday 執行時，起點與第一天漲跌停為估算；最佳委買價沒有獨立時間戳，以 lastUpdated 代理。",
    "持有期間除息參考價依「前收 − 股利」逐路徑推算，取整方式未核實（勘誤 E1）。",
    "費用取整在核實前屬未確認假設。",
    "浮點運算在不同 CPU／作業系統上可能有極小差異；core 已四捨五入以降低影響，價位運算全部使用精確的整數分，但不保證跨平台逐位元一致。",
]


def _round_core(obj):
    if isinstance(obj, dict):
        return {str(k): _round_core(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_core(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Decimal):
        return str(obj)
    return obj


def _quantize(res):
    """core 輸出：機率 1e-6、金額整數元（依欄位名稱判斷）。"""
    money_keys = ("ev", "es", "value", "pnl", "loss", "M_tp", "M_non", "B")
    def q(k, v):
        if isinstance(v, dict):
            return {kk: q(kk, vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [q(k, x) for x in v]
        if isinstance(v, float):
            kl = str(k)
            if any(kl.startswith(m) or ("_" + m) in kl for m in money_keys) or kl in ("mean_value",):
                return round(v)
            if "price" in kl:
                return round(v, 4)
            return round(v, 6)
        return v
    return {k: q(k, v) for k, v in res.items()}


def map_future_events(events, cal, today, D):
    """§4.9 持有期間事件 → 模擬交易日索引。
    - 只要有任何 ex_date 晚於今天的事件，就必須提供官方交易日曆，且日曆至少涵蓋 D 個交易日；
      不足、空白、未排序、重複或起始日不晚於今天 → 拒絕（不得用不完整日曆推定事件不存在）。
    - 模擬期間 ＝ cal[0] ～ cal[D-1]（第 0 個模擬交易日 ＝ 下一個交易日）。
    - 期間內的股票股利、減資 → 拒絕；現金股利 → 對應索引（同日多筆照列，引擎加總）。
    - 期間外的事件不影響模擬。
    回傳 (ex_days, ex_div_u, flags)。"""
    flags = []
    fut = []
    for e in events:
        for f in ("type", "ex_date"):
            if f not in e or e[f] in (None, ""):
                raise EngineRefusal(f"公司事件資料缺少 {f}：{e}")
        if e["type"] not in ("cash_dividend", "stock_dividend", "capital_reduction"):
            raise EngineRefusal(f"無法辨識的公司事件類型：{e['type']}")
        exd = _dt.date.fromisoformat(str(e["ex_date"]))
        if exd > today:
            fut.append((exd, e))
    if not fut:
        return [], [], flags
    if cal is None or len(cal) == 0:
        raise EngineRefusal("有晚於今天的公司事件，但未提供官方交易日曆（或日曆為空），無法判斷是否落在模擬期間，拒絕執行")
    cdates = [_dt.date.fromisoformat(str(x)) for x in cal]
    if any(cdates[i] >= cdates[i + 1] for i in range(len(cdates) - 1)):
        raise EngineRefusal("交易日曆必須嚴格遞增且不得重複")
    if cdates[0] <= today:
        raise EngineRefusal("交易日曆第一天必須是今天之後的下一個交易日")
    if len(cdates) < D:
        raise EngineRefusal(f"交易日曆只有 {len(cdates)} 個交易日，不足以涵蓋模擬期間 {D} 個交易日，拒絕執行")
    horizon = cdates[:D]
    end = horizon[-1]
    ex_days, ex_div_u = [], []
    for exd, e in fut:
        if exd > end:
            continue
        if e["type"] == "stock_dividend":
            raise EngineRefusal("持有期間含股票股利除權，超出本版範圍，需人工處理")
        if e["type"] == "capital_reduction":
            raise EngineRefusal("持有期間含減資事件，超出本版範圍")
        if exd not in horizon:
            raise EngineRefusal(f"除息日 {exd} 不在提供的交易日曆中（不是交易日），拒絕執行")
        raw = e.get("cash_per_share")
        if raw is None:
            raise EngineRefusal("股利資料不完整（缺每股現金股利）")
        try:
            amt = Decimal(str(raw))
        except Exception:
            raise EngineRefusal(f"每股現金股利無法解析：{raw}")
        if not amt.is_finite() or amt < 0:
            raise EngineRefusal(f"每股現金股利必須是非負數：{raw}")
        q = amt.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        if q != amt:
            flags.append(f"每股現金股利 {raw} 已四捨五入到 0.000001 元（{q}）")
        ex_days.append(horizon.index(exd))
        ex_div_u.append(int(q * 1_000_000))
    return ex_days, ex_div_u, flags


def run(inp: dict) -> dict:
    """完整執行（§1～§5）。拒絕情況拋出 EngineRefusal。"""
    mode_s = inp["mode"]
    if mode_s not in ("holding", "entry"):
        raise EngineRefusal("mode 必須是 holding 或 entry")
    mode = 0 if mode_s == "holding" else 1
    phase = inp["run_phase"]
    if phase not in ("intraday", "after_close"):
        raise EngineRefusal("run_phase 必須是 intraday 或 after_close")
    plan = inp["plan"]
    params = inp["params"]
    snap = inp["market_snapshot"]
    costs = Costs(inp["costs"])
    flags = list(costs.flags())
    if not inp.get("price_limit_normal", False):
        raise EngineRefusal("無漲跌幅限制或非一般 ±10% 規則的股票（price_limit_normal 未確認），拒絕執行")
    P = int(params["path_count"])
    seed = int(params["seed"])
    ext_cap = int(params.get("extension_cap", 10))
    fresh_min = float(params.get("freshness_minutes", 5))
    if P < 100_000:
        raise EngineRefusal("路徑數必須 ≥ 100,000")
    if ext_cap != 10:
        raise EngineRefusal("延長上限必須是 10")
    Hd = int(plan["Hd"])
    if Hd <= 0:
        raise EngineRefusal("Hd 必須為正整數")
    # 報價新鮮度
    q_at = _dt.datetime.strptime(snap["queried_at"][:19], "%Y-%m-%d %H:%M:%S")
    today = q_at.date()
    lt = snap.get("last_trade_at")
    if lt is None:
        raise EngineRefusal("缺少 last_trade_at，無法確認報價新鮮度")
    lt_dt = _dt.datetime.fromtimestamp(int(lt) / 1e6, tz=_dt.timezone(_dt.timedelta(hours=8))).replace(tzinfo=None)
    if lt_dt.date() != today:
        raise EngineRefusal(f"最後成交日 {lt_dt.date()} 不是今天 {today}（今天沒有成交），拒絕執行")
    age_min = (q_at - lt_dt).total_seconds() / 60
    if phase == "intraday" and age_min > fresh_min:
        flags.append(f"最新成交距查詢已超過 {age_min:.1f} 分鐘，起始價可能過期")
    if phase == "after_close":
        if snap.get("official_close") is None:
            raise EngineRefusal("after_close 需要官方收盤價")
        start_c = d2c(snap["official_close"])
    else:
        start_c = d2c(snap["last_price"])
        flags.append("intraday：隔日漲跌停價為估算，非正式基準")
    # 價位檢查
    tp_c = d2c(plan["take_profit"])
    stop_c = d2c(plan["stop_trigger"])
    stop_type = {"intraday": 0, "close_confirm": 1}[plan["stop_type"]]
    for nm, v in (("停利價", tp_c), ("停損／出場觸發價", stop_c)):
        if floor_legal_i(v) != v:
            raise EngineRefusal(f"{nm} {c2d(v)} 不是合法價位")
    stock_rows = check_candles(inp["raw_candles"], "個股日K")
    index_rows = check_candles(inp["raw_index_candles"], "IX0001日K")
    volume_unit_check(stock_rows, "個股日K")
    # 歷史參考價、未來事件
    hist_refs = {}
    for e in inp.get("sample_events", []):
        dd = _dt.date.fromisoformat(e["date"])
        hist_refs[dd] = Decimal(str(e["reference_price"])) if e.get("reference_price") is not None else None
    official_close_c = d2c(snap["official_close"]) if snap.get("official_close") is not None else None
    samples, dlog = build_samples(stock_rows, index_rows, phase, today, official_close_c, hist_refs)
    run_w, skip_w = plan_windows(len(samples))
    subs = [0, 0, 0, 0]
    lp = [0, 0]
    E = 0
    b2_after = 0
    fill_lo = fill_hi = 0.0
    if mode == 0:
        shares = int(plan["shares"])
        if shares <= 0:
            raise EngineRefusal("股數必須為正整數")
        if not (tp_c > start_c > stop_c):
            raise EngineRefusal(f"holding：需 停利價 {c2d(tp_c)} > 起始前收 {c2d(start_c)} > 停損／出場觸發價 {c2d(stop_c)}")
        bid = snap.get("best_bid")
        if bid is None or Decimal(str(bid)) <= 0:
            raise EngineRefusal("最佳委買價必須存在且 > 0")
        bid_c = d2c(bid)
        B = costs.sell_net(bid_c, shares)
    else:
        E = int(plan["E"])
        if E <= 0:
            raise EngineRefusal("E 必須為正整數")
        batches = plan["batches"]
        if not (1 <= len(batches) <= 2):
            raise EngineRefusal("新倉需一或兩批")
        for bi, b in enumerate(batches):
            lpc = d2c(b["limit"])
            if floor_legal_i(lpc) != lpc:
                raise EngineRefusal(f"第 {bi+1} 批限價 {b['limit']} 不是合法價位")
            if not (stop_c < lpc < tp_c):
                raise EngineRefusal(f"entry：需 停損 {c2d(stop_c)} < 第 {bi+1} 批限價 {c2d(lpc)} < 停利 {c2d(tp_c)}")
            lp[bi] = lpc
            w, r = split_shares(int(b["shares"]))
            if int(b["shares"]) <= 0:
                raise EngineRefusal("股數必須為正整數")
            subs[2 * bi] = w
            subs[2 * bi + 1] = r
        if len(batches) == 2:
            b2_after = {"open": 0, "after_first_fill": 1}[plan.get("batch2_activation", "open")]
        fi = inp["fill_interval"]
        fill_lo, fill_hi = float(fi[0]), float(fi[1])
        if not (0 <= fill_lo <= fill_hi <= 1):
            raise EngineRefusal("fill_interval 需滿足 0 ≤ 下限 ≤ 上限 ≤ 1")
        shares = None
        B = None
    D = E + Hd + ext_cap + 1
    # 持有期間的公司事件（§4.9；R1 修正：交易日曆必須涵蓋完整模擬期間）
    ex_days, ex_div_u, ev_flags = map_future_events(inp.get("future_events", []), inp.get("trading_calendar"), today, D)
    flags += ev_flags
    # 抽樣
    U, V = make_uniforms(seed, P, D, need_V=(mode == 1))
    versions = []
    win_meta = {}
    for w in run_w:
        win = samples[-w:]
        st = sample_stats(win)
        idx = window_index(U, w)
        win_meta[str(w)] = {"window_days": w, "sampling_indices_sha256": sha256_bytes(idx.astype("<i4").tobytes()),
                            "sampling_dates_sha256": dates_hash(idx, [s["date"] for s in win]),
                            "seed": seed, "path_count": P, "sample_first": str(win[0]["date"]),
                            "sample_last": str(win[-1]["date"]), "stats": st,
                            "lock_coverage": ("模擬未涵蓋鎖死風險" if st["one_price_limit_down_days"] == 0 else "樣本含一價到底跌停日")}
        vols = np.array([s["vol"] for s in win], np.int64)
        amp_base = start_c if mode == 0 else lp[0]
        amp_c = st["avg_amplitude"] * amp_base
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            R = drift_ratios(win, f, st["mu"])
            common = dict(mode=mode, start_prev_c=start_c, hold_shares=(shares or 0), tp_c=tp_c, stop_c=stop_c,
                          stop_type=stop_type, Hd=Hd, ext_cap=ext_cap, sub_shares=subs, lp_c=lp, E=E,
                          batch2_after=b2_after, ex_days=ex_days, ex_div_u=ex_div_u)
            ver = {"window": w, "drift_keep": f}
            if mode == 0:
                o = run_core(R, vols, idx, V, fill_p=0.0, **common)
                ver["result"] = evaluate_version(o, costs, 0, B=B)
                ob = run_core(R, vols, idx, V, fill_p=0.0, stressB=1, amp_c=amp_c, **common)
                eb = evaluate_version(ob, costs, 0, B=B)
                ver["test_B"] = {"ev": eb["ev"], "amplitude_amount": amp_c / 100,
                                 "note": "指定幅度的壓力假設，不是成交價預測"}
            else:
                for lab, fp in (("fill_low", fill_lo), ("fill_high", fill_hi)):
                    o = run_core(R, vols, idx, V, fill_p=fp, **common)
                    ver[lab] = evaluate_version(o, costs, 1, subs=subs)
                    ob = run_core(R, vols, idx, V, fill_p=fp, stressB=1, amp_c=amp_c, **common)
                    eb = evaluate_version(ob, costs, 1, subs=subs)
                    ver[lab]["test_B"] = {"ev_cond": eb["ev_cond"], "ev_strategy": eb["ev_strategy"],
                                          "amplitude_amount": amp_c / 100, "note": "指定幅度的壓力假設，不是成交價預測"}
            versions.append(ver)
    stress = (stress_deterministic(costs, 0, start_c, shares, B=B) if mode == 0
              else stress_deterministic(costs, 1, None, None, lp_c=lp[0], buy_shares=int(plan["batches"][0]["shares"])))
    core = {
        "engine_version": VERSION, "spec": SPEC, "mode": mode_s, "run_phase": phase,
        "start_prev": float(start_c) / 100, "baseline_B": (str(B) if B is not None else None),
        "plan": plan, "costs": costs.describe(), "fill_interval": inp.get("fill_interval"),
        "windows_run": run_w, "windows_skipped": {str(w): f"無法完成（只有 {len(samples)} 日）" for w in skip_w},
        "usable_sample_days": len(samples), "data_log": dlog, "windows": win_meta,
        "V_column_map": {"0": "batch1_whole", "1": "batch1_odd", "2": "batch2_whole", "3": "batch2_odd"},
        "sub_orders_present": [{"col": s, "batch": 1 + s // 2, "type": "whole" if s % 2 == 0 else "odd",
                                "shares": subs[s], "limit": float(lp[s // 2]) / 100} for s in range(4) if subs[s] > 0],
        "uniform_U_sha256": sha256_bytes(U.astype("<f8").tobytes()),
        "uniform_V_sha256": (sha256_bytes(V.astype("<f8").tobytes()) if V is not None else None),
        "future_cash_dividend_days": [{"sim_day": d, "cash_per_share": str(Decimal(u) / 10**6)} for d, u in zip(ex_days, ex_div_u)],
        "future_cash_dividend_by_day": {str(d): str(sum(Decimal(u) for dd, u in zip(ex_days, ex_div_u) if dd == d) / 10**6) for d in sorted(set(ex_days))},
        "versions": [_quantize(_round_core(v)) for v in versions],
        "deterministic_stress": stress, "flags": flags, "tax_note": TAX_NOTE, "model_limits": LIMITS,
        "B_note": "立即出清基準以最佳委買價估算，未考慮委買深度、分價成交及市場衝擊；零股以整股最佳委買價代理。",
    }
    core = _round_core(core)
    with open(os.path.abspath(__file__), "rb") as fh:
        prog_hash = sha256_bytes(fh.read())
    meta = {
        "program_sha256": prog_hash, "numpy_version": np.__version__, "numba": _NUMBA,
        "python": sys.version.split()[0], "run_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "input_sha256": {k: sha256_bytes(canon_json(inp[k])) for k in ("raw_candles", "raw_index_candles")},
        "queried_at": snap.get("queried_at"),
    }
    for k in ("raw_dividends", "sample_events", "future_events", "trading_calendar"):
        if k in inp:
            meta["input_sha256"][k] = sha256_bytes(canon_json(inp[k]))
    return {"core": core, "core_sha256": sha256_bytes(canon_json(core)), "meta": meta}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python3 uep_sim.py input.json output.json")
        sys.exit(2)
    with open(sys.argv[1], encoding="utf-8") as fh:
        inp = json.load(fh)
    try:
        res = run(inp)
    except EngineRefusal as e:
        res = {"refused": True, "reason": str(e)}
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({"refused": res.get("refused", False), "core_sha256": res.get("core_sha256")}, ensure_ascii=False))
