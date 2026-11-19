"""每次對話使用引擎前的快速自檢：已知答案案例（人工推導，與 tests/ 相同），任何一項不符就 exit 1，不得使用引擎。
完整測試：python3 -m pytest -q tests/"""
import sys
from decimal import Decimal as Dm
import numpy as np
import uep_sim as u

C = {"commission_rate": "0.001425", "discount": "1", "min_fee_whole": "20", "min_fee_odd": "1", "tax_rate": "0.003",
     "fee_rounding": "ROUND_FLOOR", "tax_rounding": "ROUND_FLOOR", "fee_rounding_verified": False, "tax_rounding_verified": False}
COST = u.Costs(C)
fails = []


def check(name, got, want):
    if got != want:
        fails.append(f"{name}: got {got!r}, want {want!r}")


def path(days, prev, **kw):
    prev_d, R = Dm(str(prev)), []
    for bar in days:
        R.append([float(Dm(str(x)) / prev_d) for x in bar])
        prev_d = Dm(str(bar[3]))
    D = len(days)
    V = kw.pop("V", None)
    return u.run_core(np.array(R), np.full(D, 10**9), np.arange(D).reshape(1, D), V, start_prev_c=u.d2c(prev), **kw)


# 1. 漲跌停（calc_limit_prices 標準答案）
for prev, dn, up in (("11.1", "9.99", "12.2"), ("45.5", "40.95", "50"), ("610", "549", "671"),
                     ("1100", "990", "1210"), ("52.3", "47.1", "57.5")):
    check(f"limits {prev}", u.limits_dec(prev), (Dm(dn), Dm(up)))
    check(f"limits_micro {prev}", u.limits_micro(u.d2c(prev) * 10_000), (u.d2c(dn), u.d2c(up)))
# 2. 升降單位與等距取高
check("tick_down 1000", u.tick_down_i(100000), 99900)
check("tick_up 1000", u.tick_up_i(100000), 100500)
check("round 999.5", u.round_legal_f(99950.0), 100000)
check("round 10.025", u.round_legal_f(1002.5), 1005)
# 3. 連續跌停到延長上限（§7 測試 6）
chain = [90, 81, 72.9, 65.7, 59.2, 53.3, 48.0, 43.2, 38.9, 35.05, 31.55]
o = path([(x,) * 4 for x in chain], 100, mode=0, hold_shares=1000, tp_c=11000, stop_c=8000, stop_type=1, Hd=1)
r = u.evaluate_version(o, COST, 0, B=COST.sell_net(10000, 1000))
check("T6", (int(o["cls"][0]), int(o["exit_px"][0]), int(o["cons_px"][0]), r["ev"], r["ev_conservative_unfinished"]),
      (u.CLS_UNF, 3155, 2840, -68146.0, -71283.0))
# 4. 兩批成交、成交日不停利、次日一起停利（§7 測試 49）
o = path([(605, 606, 599, 600), (600, 625, 589, 622), (618, 625, 615, 620)], 610, mode=1, tp_c=62000, stop_c=56000,
         stop_type=1, Hd=5, sub_shares=(1000, 0, 1000, 0), lp_c=(60000, 59000), E=3, fill_p=0.5,
         V=np.full((1, 3, 4), 0.99))
u.evaluate_version(o, COST, 1, subs=[1000, 0, 1000, 0])
check("T49", (int(o["cls"][0]), int(o["exit_day"][0]), int(o["_V_main_u"][0])), (u.CLS_TP, 2, 42818 * 10**6))
# 5. 盤中停損：max(tick_down(基準), L, DN)（§7 測試 20）
for bar, px in (((590, 592, 580, 585), 58000), ((590, 592, 570, 575), 57900), ((575, 578, 570, 576), 57400)):
    o = path([bar], 600, mode=0, hold_shares=1000, tp_c=65000, stop_c=58000, stop_type=0, Hd=5)
    check(f"T20 {bar}", int(o["exit_px"][0]), px)
# 6. 現金股利只計一次（§7 測試 59）
o = u.run_core(np.ones((1, 4)), np.full(1, 10**9), np.zeros((1, 2)), None, mode=0, start_prev_c=61000,
               hold_shares=1000, tp_c=70000, stop_c=50000, stop_type=1, Hd=2, ex_days=(0,), ex_div_u=(5_000_000,))
r = u.evaluate_version(o, COST, 0, B=COST.sell_net(61000, 1000))
check("T59", (int(o["exit_px"][0]), r["ev"]), (60500, 22.0))
# 7. 費用拆單與取整（§7 測試 24、46）
check("T24", COST.sell_net(5000, 1450), Dm(72180))
hu = u.Costs({**C, "fee_rounding": "ROUND_HALF_UP", "tax_rounding": "ROUND_HALF_UP"})
check("T46", (COST.sell_net(2105, 1500), hu.sell_net(2105, 1500)), (Dm(31438), Dm(31435)))
# 8. ES 的 k = ceil(0.05N)
check("ES k", u.es_mean(np.arange(101, dtype=np.int64) * 10**6), 2.5)
# 9. 共同亂數可重現
U1, _ = u.make_uniforms(7, 1000, 5, False)
U2, _ = u.make_uniforms(7, 1000, 5, False)
check("U repeat", bool(np.array_equal(U1, U2)), True)

print(f"engine {u.VERSION}  numba={u._NUMBA}  numpy={np.__version__}")
if not u._NUMBA:
    fails.append("numba 不可用：純 Python 版本跑 10 萬路徑過慢，不得用於正式執行")
if fails:
    print("SELFCHECK FAILED")
    print("\n".join(fails))
    sys.exit(1)
print("SELFCHECK OK")
