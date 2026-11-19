"""突變測試：對引擎注入單一規則錯誤，確認測試套件會失敗（證明測試有偵測力）。
用法：python3 tools/mutation_check.py"""
import os, shutil, subprocess, sys, tempfile
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUTANTS = [
    ("M1 停利優先於停損", "if sl_ok:\n                    if lock:", "if sl_ok and not tp_ok:\n                    if lock:"),
    ("M2 停損成交價不扣 1 tick", "base = O if O < stop_c else stop_c\n                        pxs = max(tick_down_i(base), L, dn)",
     "base = O if O < stop_c else stop_c\n                        pxs = max(base, L, dn)"),
    ("M3 成交當日停利不封鎖", "if (not fill_today) and (not lock):", "if (not lock):"),
    ("M4 移除成交量 1% 前提", "volok = ss * 100 <= vols[j]", "volok = True"),
    ("M5 同日股利覆蓋（R2 舊錯）", "dk += ex_div_u[k]", "dk = ex_div_u[k]"),
    ("M6 LOCK 不檢查最高價", "(H == dn) and (L == dn)", "(L == dn)"),
    # 註：拿掉「C == DN」是等價突變（K 線合法時 O=H=L=DN 必然 C=DN），不列入。
    ("M7 統一期限差一日", "d >= first_fill + Hd - 1:", "d >= first_fill + Hd:"),
    ("M8 E 有效天數差一日", "if d >= sub_start[s] + E - 1:", "if d >= sub_start[s] + E:"),
    ("M9 ES 的 k 用 floor", "k = -(-N // 20)", "k = max(1, N // 20)"),
    ("M10 待執行賣出用當日 L", "pxs = max(tick_down_i(O), L, dn)\n                    if stressB", "pxs = max(L, dn)\n                    if stressB"),
    ("M11 測試 B 受 L 限制", "pxs = max(round_legal_f(pxs - amp_c), dn)\n                        state = ST_EXITED\n                        o_cls[p] = CLS_SL",
     "pxs = max(round_legal_f(pxs - amp_c), dn, L)\n                        state = ST_EXITED\n                        o_cls[p] = CLS_SL"),
    ("M12 剛好觸及不需 V", "elif L == Lp:\n                        touched[s] = 1\n                        if V[p, d, s] < fill_p:",
     "elif L == Lp:\n                        touched[s] = 1\n                        if True:"),
    ("M13 量不足時以 O 成交", "filled = True\n                            fpx = Lp\n                            kind = K_VOL_V\n                    elif L <= tick_down_i(Lp):",
     "filled = True\n                            fpx = O\n                            kind = K_VOL_V\n                    elif L <= tick_down_i(Lp):"),
    ("M14 除息日成交也取得股利", "if (state == ST_HELD or state == ST_PENDING) and shares > 0:\n                    div += shares * dk",
     "if True:\n                    div += (shares + sub_shares[0]) * dk"),
    ("M15 日曆過短仍接受（R1 舊錯）", "if len(cdates) < D:", "if len(cdates) < 1:"),
    ("M16 收盤確認在 LOCK 日不觸發", "elif stop_type == 1 and C < stop_c:", "elif stop_type == 1 and C < stop_c and not lock:"),
    ("M17 保守估值未再跌停", "o_cons_px[p] = cdn", "o_cons_px[p] = dn"),
    ("M18 手續費合併計算", "for sh in ((k * 1000) if k else 0, rem):\n                if sh > 0:\n                    amt, fee, tax = self.order(px_c, sh, \"sell\")",
     "for sh in ([total_shares]):\n                if sh > 0:\n                    amt, fee, tax = self.order(px_c, sh, \"sell\") if sh < 1000 or sh % 1000 == 0 else self.order(px_c, (sh // 1000) * 1000, \"sell\")"),
]
def main():
    src = open(os.path.join(SIM, "uep_sim.py"), encoding="utf-8").read()
    rows, survived = [], 0
    for name, old, new in MUTANTS:
        assert src.count(old) == 1, f"{name}: 原碼片段出現 {src.count(old)} 次"
        tmp = tempfile.mkdtemp()
        shutil.copytree(os.path.join(SIM, "tests"), os.path.join(tmp, "tests"))
        open(os.path.join(tmp, "uep_sim.py"), "w", encoding="utf-8").write(src.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider",
                            "tests/test_spec_v5.py", "tests/test_regression.py"], cwd=tmp, capture_output=True, text=True)
        killed = r.returncode != 0
        first = next((l for l in r.stdout.splitlines() if l.startswith("FAILED")), "")
        rows.append((name, "偵測到" if killed else "未偵測（存活）", first.split(" - ")[0].replace("FAILED ", "")))
        survived += (not killed)
        shutil.rmtree(tmp)
    for r_ in rows:
        print(" | ".join(r_))
    print(f"mutants={len(MUTANTS)} killed={len(MUTANTS)-survived} survived={survived}")
    sys.exit(1 if survived else 0)
main()
