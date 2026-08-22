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

MODELS := m1_base_up20 m2_nomkt_up20 m3_v3_up20 m4_v3nomkt_up20 m5_v3nomv_up20 \
          m6_base_nobear m7_nomkt_nobear m8_v3_nobear m9_v3nomkt_nobear m10_v3nomv_nobear

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
	@echo "  重建 m1~m10（最高原則：這條路徑必須永遠走得通）"
	@echo "    make bootstrap      從 $(BOOTSTRAP_FROM) 起全量抓取（10~15 小時）"
	@echo "    make rebuild-full   先刪衍生檔再全量重建特徵與 label"
	@echo "    make train          序列訓練 10 個模型 + 產門檻曲線（數小時）"
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
	@if ! (pkg-config --exists ta-lib 2>/dev/null || ls /opt/homebrew/lib/libta*.dylib \
	        /usr/local/lib/libta*.dylib /usr/lib64/libta*.so >/dev/null 2>&1); then \
	  echo ""; \
	  echo "  ❌ 找不到 TA-Lib 的 C 函式庫。先裝它再回來："; \
	  echo "       macOS         brew install ta-lib libomp"; \
	  echo "       Oracle Linux  sudo yum install ta-lib ta-lib-devel"; \
	  echo "     （0.4.29 與 Homebrew 現行 ta-lib 0.7.x 不相容，requirements 鎖 0.6.8）"; \
	  echo ""; \
	  exit 1; \
	fi
	@test -d .venv || python3 -m venv .venv
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

scores:  ## 對新日期補算 10 個模型的分數（前端歷史曲線用）
	$(PY) -m engine.models.score_recent

# ── 日常更新 ────────────────────────────────────────────────────────────
update: data revenue promote validate features labels scores  ## 日常增量更新
	@echo ""
	@echo "  全部更新完成，make app 看最新推薦"
	@echo ""

# ── 全量重建 ────────────────────────────────────────────────────────────
bootstrap:  ## 從 $(BOOTSTRAP_FROM) 起全量抓取（10~15 小時）
	$(PY) -m engine.data_source.fetch_stock_list
	$(PY) -m engine.data_source.fetch_price_official --start $(BOOTSTRAP_FROM) --end $(TODAY)
	$(PY) -m engine.data_source.fetch_price_official --start $(BOOTSTRAP_FROM) --end $(TODAY) --market index
	$(PY) -m engine.data_source.fetch_exright   --first-year $(shell echo $(BOOTSTRAP_FROM) | cut -d- -f1)
	$(PY) -m engine.data_source.fetch_chip      --start $(BOOTSTRAP_FROM) --end $(TODAY)
	$(PY) -m engine.data_source.fetch_chip_tpex --start $(BOOTSTRAP_FROM) --end $(TODAY)
	$(PY) -m engine.data_source.fetch_fundamental --start $(BOOTSTRAP_FROM) --end $(TODAY)
	$(PY) -m engine.data_source.fetch_revenue --bulk-start $(shell echo $(BOOTSTRAP_FROM) | cut -c1-7) --bulk-end $(REV_TO)
	@$(MAKE) promote
	@$(MAKE) validate

# builder 是 upsert 寫檔，--full 只覆蓋算得出來的列，舊資料獨有的組合會殘留
# （2026-08-22 踩過：2026-07-10 那個假交易日的 1,948 列留在特徵裡）。
# 所以全量重建前一定要先刪（CLAUDE.md 規則 5）。
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
train:  ## 序列訓練 m1~m10（含調參與門檻曲線，數小時）
	./engine/models/train_all.sh

# 平常不用跑。m1/m2/m6/m7 的組態來自版控的 engine/models/config/sweep_round4_rf.csv
# （2026-08-15 在 344 欄上搜出來的，重訓沿用是刻意的）。只有在你想在**新資料**上
# 重搜 base 組態時才跑這個 —— 27 組、約 15 小時，跑完寫進 data/，
# best_config() 會自動優先讀 data/ 的新版本。
sweep-base:  ## 重搜 base 特徵集的超參數（27 組，約 15 小時，平常不用）
	$(PY) -m engine.models.sweep_round1 --model rf --round 4 --features data/features.parquet

curve:  ## 產生 val_sel 門檻曲線（訓練後由人看曲線挑門檻）
	@for k in $(MODELS); do \
	  echo "=== $$k ==="; \
	  $(PY) -m engine.models.threshold_curve --tag $$k --split val_sel \
	      --trail-trigger 0.15 --trail-pct 0.10 --stop-loss 0.20; \
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
backtest:  ## 用挑定門檻回測 10 個模型（絕對門檻 + 訊號數對齊）
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
