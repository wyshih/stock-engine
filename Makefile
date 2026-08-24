.DEFAULT_GOAL := help
SHELL := /bin/bash
PY      := .venv/bin/python
TODAY   := $(shell date +%Y-%m-%d)
PORT    ?= 8501
DASHBOARD ?= ../dashboard

# 月營收：M 月的營收要到 M+1 月 10 日才公告，所以今天能抓的最新是「上個月」。
# 抓上兩個月（1~9 號時上月還沒公告，這時靠上上月墊著；重複抓是 upsert，無害）。
# 抓「本月」會打到還不存在的頁面，read_html 找不到表格再重試三次，白等兩分鐘。
REV_TO   := $(shell date -v-1m +%Y-%m 2>/dev/null || date -d '1 month ago' +%Y-%m)
REV_FROM := $(shell date -v-2m +%Y-%m 2>/dev/null || date -d '2 months ago' +%Y-%m)

# 全量重建的起點（資料期間 2019-01-02 ~）
BOOTSTRAP_FROM ?= 2019-01-01

# 不放 /tmp：macOS 會定期清空，前端掛掉時 log 常常已經跟著不見，查不到原因
APP_LOG ?= $(HOME)/Library/Logs/stock_app.log
# 區域網路 IP（macOS 先問 Wi-Fi 再問有線；取不到就退回 hostname）
LAN_IP  := $(shell ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $$1}')

MODELS := m1_base_up20 m2_nomkt_up20 m3_v3_up20 m6_base_nobear m8_v3_nobear

.PHONY: help install bootstrap update data promote revenue validate features features-v3 \
        labels scores train curve backtest app app-bg stop restart logs \
        export-public publish-public status test verify-vs-old rebuild-full clean-derived

help:  ## 列出指令
	@echo ""
	@echo "  日常"
	@echo "    make update         抓資料 → promote → 建特徵 → 算 label → 補分數"
	@echo "    make app            啟動前端  http://localhost:$(PORT)"
	@echo "    make status         看各資料檔的最新日期與斷層"
	@echo ""
	@echo "  重建這 5 個模型（最高原則：這條路徑必須永遠走得通）"
	@echo "    make bootstrap      從 $(BOOTSTRAP_FROM) 起全量抓取（10~15 小時）"
	@echo "    make rebuild-full   先刪衍生檔再全量重建特徵與 label"
	@echo "    make train          序列訓練 5 個模型 + 產門檻曲線（數小時）"
	@echo "    make curve          只產門檻曲線（由人看曲線挑門檻）"
	@echo "    make backtest       用挑定門檻回測，附訊號數對齊版"
	@echo ""
	@echo "  其他"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| grep -vE '^(help|update|app|status|bootstrap|rebuild-full|train|curve|backtest):' \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── 安裝 ────────────────────────────────────────────────────────────────
install:  ## 建立 venv、檢查 TA-Lib C 函式庫、安裝套件
	@# TA-Lib 的 Python wheel 需要先有 C 二進位，缺了會在 pip install 階段才爆，
	@# 訊息又完全看不出是缺 C library。先擋在這裡。
	@# 每個路徑分開檢查：`ls a b c` 只要其中一個不存在就回非零，而 /usr/local/lib
	@# 與 /usr/lib64 在 Apple Silicon 上本來就沒有 —— 合在一起寫會誤報找不到。
	@if ! (pkg-config --exists ta-lib 2>/dev/null \
	       || ls /opt/homebrew/lib/libta*.dylib >/dev/null 2>&1 \
	       || ls /usr/local/lib/libta*.dylib >/dev/null 2>&1 \
	       || ls /usr/lib64/libta*.so >/dev/null 2>&1); then \
	  echo ""; \
	  echo "  ❌ 找不到 TA-Lib 的 C 函式庫。先裝它再回來："; \
	  echo "       macOS         brew install ta-lib libomp"; \
	  echo "       Oracle Linux  sudo yum install ta-lib ta-lib-devel"; \
	  echo "     （0.4.29 與 Homebrew 現行 ta-lib 0.7.x 不相容，requirements 鎖 0.6.8）"; \
	  echo ""; \
	  exit 1; \
	fi
	@# 用 3.12：macOS 的系統 python3 是 3.9.6，建在上面不會馬上爆，是等到跑起來
	@# 才出現難查的問題。上限也不能太新 —— requirements 鎖的 numpy 1.26.4 沒有
	@# 3.13+ 的 wheel，硬升 numpy 會連帶升 scikit-learn，RF 的訓練結果就跟舊模型
	@# 不可比了。3.12 是「夠新且所有套件版本都不用動」的那一格。
	@if [ ! -d .venv ]; then \
	  PYBIN=$$(command -v python3.12 || command -v python3.11 || command -v python3); \
	  ver=$$($$PYBIN -c 'import sys;print("%d%02d"%sys.version_info[:2])'); \
	  if [ "$$ver" -lt 311 ]; then \
	    echo ""; echo "  ❌ 需要 Python 3.11 以上，找到的是 $$($$PYBIN -V)"; \
	    echo "       macOS  brew install python@3.12"; echo ""; exit 1; \
	  fi; \
	  echo "  用 $$PYBIN（$$($$PYBIN -V)）建立 venv"; \
	  $$PYBIN -m venv .venv; \
	fi
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@# fetch_stock_list 是 bootstrap/update 的第一步且會 load_dotenv()，
	@# 沒有 .env 時 token 會靜默變空字串，清單抓回來是空的、整條流程白跑。
	@test -f .env || { cp .env.example .env; \
	  echo ""; echo "  ⚠️ 已從 .env.example 建立 .env —— 請填入 FINMIND_TOKEN 再繼續"; }
	@echo ""
	@echo "  完成。接著 make bootstrap（第一次）或 make update（日常）"
	@echo ""

# ── 資料 ────────────────────────────────────────────────────────────────
# fetch_stock_list 一定要最前面：抓價格是照 stock_list 逐檔對的，
# 不先更新清單，新上市的股票永遠不會有資料（CLAUDE.md 規則 12）。
# 用區間模式（從資料最後一天接著抓到今天）而不是只抓當天 —— 只抓當天的話，
# 沒天天跑就會留洞（2026-08-14 發現 8/3~8/12 共 8 個交易日全缺）。
data:  ## 只抓資料（自動從缺漏處補到今天）
	$(PY) -m engine.data_source.fetch_stock_list
	@from=$$($(PY) -m engine.data_source.fetch_from); \
	echo "抓取區間：$$from ~ $(TODAY)"; \
	$(PY) -m engine.data_source.fetch_price_official --start $$from --end $(TODAY) && \
	$(PY) -m engine.data_source.fetch_price_official --start $${from:0:8}01 --end $(TODAY) --market index && \
	$(PY) -m engine.data_source.fetch_exright   --first-year $$(date +%Y) && \
	$(PY) -m engine.data_source.fetch_chip      --start $$from --end $(TODAY) && \
	$(PY) -m engine.data_source.fetch_chip_tpex --start $$from --end $(TODAY) && \
	$(PY) -m engine.data_source.fetch_fundamental --start $$from --end $(TODAY)

# 用 bulk 模式（整月彙總頁，一個月 2 個請求）而不是逐支股票的 --date 模式：
# 逐支要跑約 100 分鐘，bulk 抓兩個月只要 15 秒，而且是 upsert，天天跑也不會重複。
# 抓「上兩個月」不是「本月」—— 本月的頁面還不存在。
revenue:  ## 抓月營收（近兩個月，upsert）
	$(PY) -m engine.data_source.fetch_revenue --bulk-start $(REV_FROM) --bulk-end $(REV_TO)

# price_official.parquet → price.parquet。下游全部讀後者，這一步不能跳。
promote:  ## price_official → price（下游唯一讀的那份）
	$(PY) -m engine.data_source.promote_price

# 驗「資料裡最新的那一天」而不是「今天」：開盤前跑的話今天還沒有資料，
# 拿今天去驗必定失敗，整條 update 會在這裡中斷（2026-08-14 早上實際踩到）。
validate:  ## 驗證最新一個交易日的資料
	$(PY) -m engine.data_source.validate_data --date $$($(PY) -m engine.data_source.fetch_from --last)

# ── 特徵 / label ────────────────────────────────────────────────────────
# 順序不可換：build_features 是合併步驟，必須最後跑（CLAUDE.md 規則 12）。
features:  ## 增量建特徵（features.parquet，380 欄）
	$(PY) -m engine.features.build_price_features
	$(PY) -m engine.features.build_chip_features
	$(PY) -m engine.features.build_fundamental_features
	$(PY) -m engine.features.build_talib_features
	$(PY) -m engine.features.build_swing_features
	$(PY) -m engine.features.build_market_features
	$(PY) -m engine.features.build_trendline_features
	$(PY) -m engine.features.build_relative_features
	$(PY) -m engine.features.build_features

# build_v3_features.py 拒絕直接寫進 data/（產出必須先落在暫存），所以先產再搬。
features-v3:  ## 建 v3 特徵集（features_v3.parquet，520 欄）
	$(PY) -m engine.features.v3.audit
	$(PY) -m engine.features.v3.build_v3_features \
		--audit data/feature_audit.csv --volproxy data/volproxy.csv \
		--out $${TMPDIR:-/tmp}/features_v3.parquet
	cp $${TMPDIR:-/tmp}/features_v3.parquet data/features_v3.parquet

labels:  ## 算 label（labels.parquet / labels_nobear.parquet）
	$(PY) -m engine.models.build_labels
	$(PY) -m engine.models.build_labels_nobear

scores:  ## 對新日期補算 5 個模型的分數（前端歷史曲線用）
	$(PY) -m engine.models.score_recent

# ── 日常更新 ────────────────────────────────────────────────────────────
update: data revenue promote validate features labels scores  ## 日常增量更新
	@echo ""
	@echo "  全部更新完成，make app 看最新推薦"
	@echo ""

# ── 全量重建 ────────────────────────────────────────────────────────────
bootstrap:  ## 從 $(BOOTSTRAP_FROM) 起全量抓取（依主機分流 + 兩階段，約 3~4 小時）
	@# ── 為什麼慢 ────────────────────────────────────────────────────────
	@# 官方端點（TWSE MI_INDEX / T86 / BWIBBU_d、TPEx otc）都只回「某一天的
	@# 全市場快照」，沒有任何 startDate/endDate 參數，所以是一天一個請求 ×
	@# 1,994 個交易日。2026-08-23 查過官方 API：TWSE OpenAPI 130+ 個端點、
	@# TPEx 225 個，**沒有一個帶日期參數**；data.gov.tw 只是 OpenAPI 的目錄殼，
	@# 沒有歷史包；唯一整段下載在付費 E-Shop 且是 tick 檔。唯一的「多天」端點
	@# 是個股×月（STOCK_DAY / tradingStock），換過去要 144,000 個請求，比現在
	@# 的 3,988 貴 36 倍。**現在的做法已經是請求數最少的那個**。
	@#
	@# ── 併行的兩條鐵則 ──────────────────────────────────────────────────
	@# 1. **同一台主機只能有一條連線。** 2026-08-23 踩過：按資料源切成五組，
	@#    其中四組打 www.twse.com.tw，等效間隔壓到 1 秒以下 —— 15 分鐘失敗 48 次
	@#    （單執行緒時是 8 小時 33 次），出現 502 與 HTML 錯誤頁，最後被 WAF
	@#    擋成 307（只有近三天的快取日期還回得了 200）。停止後約 1 分鐘解除。
	@#    同日的 API 調查也實測到 2~3 秒間隔打 twse 約 6 個請求就會被擋；
	@#    TPEx 側則完全沒有節流。
	@# 2. **寫同一個 parquet 的不能同時跑。**
	@#      price_official 與 --market index → price_official.parquet
	@#      fetch_chip 與 fetch_chip_tpex     → chip.parquet
	@#
	@# 兩條鐵則合起來就是下面的兩階段：
	@#   階段一  lane TWSE：上市行情 → 大盤 → 除權息 → 上市籌碼（序列，同主機）
	@#           lane TPEx：上櫃行情（不同主機，與上面併行）
	@#           lane MOPS：月營收（不同主機，全程併行）
	@#   上市／上櫃行情分成兩個 lane 是跟舊 repo 學的：`logs/backfill_twse.log` 與
	@#   `backfill_tpex.log` 的第一行時間戳同一秒，證實它當初就是並行跑的
	@#   —— TWSE 1h55m（3.0s/天）、TPEx 2h27m（4.0s/天），並行後總計 2h27m。
	@#   本專案 2026-08-23 第一次回補用了預設的 `--market both`（同一行程內先打
	@#   TWSE 再打 TPEx，每天 8 秒），花了 4.4 小時 —— 整整慢一倍。
	@#   分開抓會產生三個檔，最後由 merge_price_sources 合併回
	@#   price_official.parquet（舊 repo 這步是手動做的、沒留下程式）。
	@#   階段二  lane TWSE：財報          ⎫ 不同主機、不同輸出檔，
	@#           lane TPEx：上櫃籌碼      ⎭ 可以併行
	@#   上櫃籌碼排在階段二，是因為它與上市籌碼寫同一個 chip.parquet（鐵則 2），
	@#   而不是因為主機衝突 —— 它打的是 tpex.org.tw。
	@#
	@# 中斷後直接重跑本 target：backfill.sh 的標記檔會讓已完成的段直接跳過。
	$(PY) -m engine.data_source.fetch_stock_list
	@set -m; \
	$(PY) -m engine.data_source.fetch_revenue --bulk-start $(shell echo $(BOOTSTRAP_FROM) | cut -c1-7) --bulk-end $(REV_TO) \
	  > logs/bs_revenue.log 2>&1 & pr=$$!; \
	( BACKFILL_TAG=twse engine/data_source/backfill.sh engine.data_source.fetch_price_official $(BOOTSTRAP_FROM) $(TODAY) --market twse --out price_official_twse \
	  && $(PY) -m engine.data_source.fetch_price_official --start $(BOOTSTRAP_FROM) --end $(TODAY) --market index --out price_official_index \
	  && $(PY) -m engine.data_source.fetch_exright --first-year $(shell echo $(BOOTSTRAP_FROM) | cut -d- -f1) \
	  && engine/data_source/backfill.sh engine.data_source.fetch_chip $(BOOTSTRAP_FROM) $(TODAY) \
	) > logs/bs_twse.log 2>&1 & pt=$$!; \
	BACKFILL_TAG=tpex engine/data_source/backfill.sh engine.data_source.fetch_price_official $(BOOTSTRAP_FROM) $(TODAY) --market tpex --out price_official_tpex \
	  > logs/bs_tpex_price.log 2>&1 & pq=$$!; \
	echo "  階段一　TWSE=$${pt} 上櫃行情=$${pq} 月營收=$${pr}"; \
	if wait $${pt}; then echo "  ✓ 階段一 TWSE 完成（上市行情/大盤/除權息/上市籌碼）"; \
	else echo "  ✗ 階段一 TWSE 失敗（見 logs/bs_twse.log）"; exit 1; fi; \
	if wait $${pq}; then echo "  ✓ 階段一 上櫃行情完成"; \
	else echo "  ✗ 上櫃行情失敗（見 logs/bs_tpex_price.log）"; exit 1; fi; \
	$(PY) -m engine.data_source.merge_price_sources || exit 1; \
	engine/data_source/backfill.sh engine.data_source.fetch_fundamental $(BOOTSTRAP_FROM) $(TODAY) \
	  >> logs/bs_twse.log 2>&1 & pf=$$!; \
	engine/data_source/backfill.sh engine.data_source.fetch_chip_tpex $(BOOTSTRAP_FROM) $(TODAY) \
	  > logs/bs_tpex.log 2>&1 & pp=$$!; \
	echo "  階段二　財報(TWSE)=$${pf}　上櫃籌碼(TPEx)=$${pp}　併行中"; \
	rc=0; \
	if wait $${pf}; then echo "  ✓ 財報完成"; else echo "  ✗ 財報失敗（見 logs/bs_twse.log）"; rc=1; fi; \
	if wait $${pp}; then echo "  ✓ 上櫃籌碼完成"; else echo "  ✗ 上櫃籌碼失敗（見 logs/bs_tpex.log）"; rc=1; fi; \
	if wait $${pr}; then echo "  ✓ 月營收完成"; else echo "  ✗ 月營收失敗（見 logs/bs_revenue.log）"; rc=1; fi; \
	exit $$rc
	@$(MAKE) promote
	@$(MAKE) validate

clean-derived:  ## 刪掉所有衍生檔（特徵 / label / 分數 / 曲線）
	@echo "  即將刪除 data/ 底下的衍生檔（原始資料不動）"
	@rm -fv data/{price,chip,fundamental,talib,swing,market,trendline,relative,revenue}_features.parquet
	@rm -fv data/features.parquet data/features_v3.parquet
	@rm -fv data/labels.parquet data/labels_nobear.parquet
	@rm -fv data/feature_audit.csv data/volproxy.csv
	@rm -fv data/score_*.parquet data/sigcurve_*.csv data/threshold_curve_*

rebuild-full: clean-derived features features-v3 labels  ## 先刪衍生檔再全量重建
	@echo ""
	@echo "  重建完成。接著 make train（數小時）"
	@echo ""

# ── 訓練 ────────────────────────────────────────────────────────────────
# 一次一個，不並行 —— 10 核機器，RF 內層已吃 6 核，並行只會更慢（CLAUDE.md 規則 10）。
train:  ## 序列訓練 5 個模型（含 v3 調參與門檻曲線，數小時）
	./engine/models/train_all.sh

# 平常不用跑。m1/m2/m6/m7 的組態來自版控的 engine/models/config/sweep_round4_rf.csv
# （2026-08-15 在 344 欄上搜出來的，重訓沿用是刻意的）。只有在你想在**新資料**上
# 重搜 base 組態時才跑這個 —— 27 組、約 15 小時，跑完寫進 data/，
# best_config() 會自動優先讀 data/ 的新版本。
sweep-base:  ## 重搜 base 特徵集的超參數（27 組，約 15 小時，平常不用）
	$(PY) -m engine.models.sweep_round1 --model rf --round 4 --features data/features.parquet

# 出場參數刻意不傳：threshold_curve 的預設值＝backtest.py 的 CURRENT_EXIT_RULES
# （唯一來源）。這裡再寫一次字面值等於多一份會漂移的副本 —— 2026-08-24 之前
# EXIT_DEFAULTS 就已經漂到 trail_trigger=0.25 / stop_loss=None，只因為這行覆寫
# 才沒把錯誤的曲線跑出來。要改出場規則，改 CURRENT_EXIT_RULES 一個地方。
curve:  ## 產生 val_sel 門檻曲線（訓練後由人看曲線挑門檻）
	@for k in $(MODELS); do \
	  echo "=== $$k ==="; \
	  $(PY) -m engine.models.threshold_curve --tag $$k --split val_sel; \
	  cp data/threshold_curve_$${k}_val_sel.csv data/sigcurve_$${k}_val_sel.csv; \
	done
	@echo ""
	@echo "  ⚠️ 門檻由人看曲線挑，不用自動規則；挑完寫進 engine/models/bundle.py"
	@echo "     的 CHOSEN_THRESHOLDS。每次重訓都要重挑。"
	@echo ""

# 回測的唯一路徑。simulate()/performance() + CURRENT_EXIT_RULES + dedup=False，
# 並且一定附訊號數對齊版（每日前 1.5%）—— 本系統「訊號越少報酬越高」，
# 只比固定門檻會退化成比門檻鬆緊（BACKTEST_LOG #28、CLAUDE.md 規則 9）。
# 區間預設 test + test2（Round 4 樣本外全段），不跟 public 展示窗口綁在一起。
# 要縮區間：make backtest ARGS="--start 2026-01-01"
backtest:  ## 用挑定門檻回測 5 個模型（絕對門檻 + 訊號數對齊）
	@mkdir -p data/backtest
	$(PY) -m engine.backtest.summary --out data/backtest $(ARGS)
	@column -s, -t data/backtest/backtest_summary.csv
	@echo ""
	@echo "  ⚠️ 跑完立刻寫 doc/BACKTEST_LOG.md（CLAUDE.md 規則 14）"
	@echo ""

# ── 前端 ────────────────────────────────────────────────────────────────
app:  ## 啟動前端（佔住終端機，Ctrl-C 結束）
	@echo ""
	@echo "  本機      http://localhost:$(PORT)"
	@echo "  同網域    http://$(LAN_IP):$(PORT)"
	@echo ""
	@$(PY) -m streamlit run engine/app/streamlit_app.py \
		--server.address 0.0.0.0 --server.port $(PORT) --server.headless true

app-bg:  ## 背景啟動前端（不佔終端機）
	@nohup $(PY) -m streamlit run engine/app/streamlit_app.py \
		--server.address 0.0.0.0 --server.port $(PORT) --server.headless true \
		> $(APP_LOG) 2>&1 & \
	sleep 3; \
	echo ""; echo "  http://localhost:$(PORT)　　log: make logs"; echo ""

# kill 之後要等 port 真的釋放：`make restart` 是 stop 接 app-bg，不等的話
# 新行程會在舊的還沒死透時啟動，直接吐 "Port 8501 is already in use" 然後結束。
stop:  ## 關閉前端
	@pid=$$(lsof -ti tcp:$(PORT) 2>/dev/null); \
	if [ -z "$$pid" ]; then echo "port $(PORT) 沒有執行中的前端"; exit 0; fi; \
	kill $$pid; \
	for i in $$(seq 1 20); do \
		lsof -ti tcp:$(PORT) >/dev/null 2>&1 || { echo "已關閉 port $(PORT) 的前端"; exit 0; }; \
		sleep 0.5; \
	done; \
	echo "port $(PORT) 10 秒後仍被佔用，改用 kill -9"; \
	kill -9 $$(lsof -ti tcp:$(PORT) 2>/dev/null) 2>/dev/null; \
	sleep 1; echo "已強制關閉"

restart: stop app-bg  ## 重啟前端

logs:  ## 看前端 log
	@tail -f $(APP_LOG)

# ── public dashboard ────────────────────────────────────────────────────
# 匯出後**一定要**跑 dashboard 的 pytest：public repo 的安全檢查（無 .pkl、
# 無 2025-02-01 之前的資料、無 .env）在 public_data/ 是空的時候會 skip，
# 只有資料包產出後那幾條才真的驗得到。
export-public:  ## 產出 public repo 的資料包到 $(DASHBOARD)/public_data（並驗安全）
	$(PY) -m engine.export.build_public_bundle --out $(DASHBOARD)/public_data
	@echo ""
	@echo "  ── 驗 public repo 的安全檢查（資料包已存在，skip 的那幾條現在會真的跑）──"
	@cd $(DASHBOARD) && $(MAKE) test

# 刻意只印指令、不自動 push —— public repo 推出去就收不回來了，
# 由人看過 git status 再決定。
publish-public: export-public  ## 產檔並印出 git push 指令（不自動 push）
	@echo ""
	@echo "  資料包已就緒。確認內容沒問題之後，自己執行："
	@echo ""
	@echo "    cd $(DASHBOARD)"
	@echo "    git status                 # ← 先看清楚要推什麼"
	@echo "    git add public_data"
	@echo "    git commit -m 'chore: 更新測試期資料包'"
	@echo "    git push"
	@echo ""
	@echo "  ⚠️ 推之前確認：沒有 .pkl、沒有 features*.parquet、"
	@echo "     沒有 2025-02 之前的資料、沒有 .env、沒有 doc/BACKTEST_LOG.md"
	@echo ""

# ── 檢查 ────────────────────────────────────────────────────────────────
status:  ## 看各資料檔的最新日期與斷層
	@$(PY) -m engine.tools.data_status

test:  ## 跑單元測試
	$(PY) -m pytest tests/ -q

verify-vs-old:  ## 對照舊 repo 的 data/（唯讀）逐項驗證重建結果
	$(PY) -m engine.tools.verify_vs_old
