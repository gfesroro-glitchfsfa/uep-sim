# uep-sim

UEP（台股交易研究 Prompt）第四步的蒙地卡羅模擬引擎。規格：`uep_sim_spec_v5.md`（2026-11-19 凍結）。

- 引擎：`uep_sim.py`（VERSION 1.0.1）
- 快速自檢：`python3 selfcheck.py`（約 5 秒，已知答案案例，失敗就不得使用）
- 完整測試：`python3 -m pytest -q tests/`（119 項，約 1 分鐘）
- 突變測試：`python3 tools/mutation_check.py`
- 測試報告：`UEP_sim_test_report_v1.0.1.md`
- 執行：`python3 uep_sim.py input.json output.json`
- 完整性：`sha256sum -c SHA256SUMS`

通過測試只代表在規格與測試條件下計算一致，不代表模擬機率準確，也不代表策略有正期望值。
本程式不含任何金鑰或個人資料；行情資料由呼叫端提供。
