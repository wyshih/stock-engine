# 環境安裝手冊

**給 Claude Code 執行：照著以下步驟幫我安裝股票分析預測軟體的開發環境。每個步驟完成後告訴我結果，遇到錯誤停下來回報。**

---

## 前提確認

先執行以下確認，任何一項失敗就停下來告訴我：

```bash
node --version      # 需要 18+
python --version    # 需要 3.9+
git --version
claude --version
```

---

## Step 1：安裝 ECC

```
/plugin marketplace add https://github.com/affaan-m/everything-claude-code
/plugin install ecc@ecc
```

完成後執行互動設定：

```
configure ecc
```

設定精靈出現時，選擇以下項目：
- 安裝位置：**User-level（~/.claude/）**
- Framework & Language：**Python**
- Workflow & Quality：**TDD、verification、security review**
- Rules：**Common rules、Python rules**

---

## Step 2：安裝 Python + ML Agents

```
/plugin marketplace add wshobson/agents
/plugin install python-development@claude-code-workflows
/plugin install machine-learning-ops@claude-code-workflows
/plugin install comprehensive-review@claude-code-workflows
/plugin install unit-testing@claude-code-workflows
```

---

## Step 3：安裝量化分析 Skills

```bash
npx -y skills add K-Dense-AI/claude-scientific-skills
```

---

## Step 4：安裝股票研究 Plugin

```
/plugin marketplace add quant-sentiment-ai/claude-equity-research
/plugin install trading-ideas@claude-equity-research-marketplace
```

---

## Step 5：建立 CLAUDE.md

在專案根目錄建立 `CLAUDE.md`，內容如下：

```markdown
## 實作規則
- 所有實作必須先讀取 PLAN.md，以它為唯一需求來源
- 每完成一個項目，在 PLAN.md 標記 [x]
- 遇到 PLAN.md 沒有涵蓋的情況，停下來問，不要自己決定
- 不要修改 scope 外的任何檔案

## Stack
- 後端：FastAPI + pandas + numpy + ta + scikit-learn / statsmodels
- 前端：Streamlit（prototyping）或 React（正式版）
- 資料：yfinance（開發）→ FMP API（正式）
- 測試：pytest + hypothesis

## Domain 規則
- 所有金融計算必須有 unit test 驗證邊界條件
- 預測模型必須有 backtesting，不能只有 forward-looking 結果
- 數據來源必須記錄在 code comment
- 不得 hardcode 任何 API key，一律用環境變數

## Commit 規範
feat / fix / refactor / test / docs
```

---

## Step 6：確認安裝結果

執行以下指令，列出已安裝的 agents 和 skills：

```
/agents
/skills
```

確認以下項目存在，全部打勾後回報給我：

- [ ] `python-pro` agent
- [ ] `fastapi-pro` agent
- [ ] `security-auditor` agent
- [ ] `ecc:tdd-guide` agent（原文件寫 `tdd-orchestrator`，實際名稱為此）
- [ ] scikit-learn skill
- [ ] machine-learning-ops commands

---

## 完成

所有步驟完成後告訴我：
1. 哪些成功安裝
2. 哪些失敗或找不到
3. 目前 `~/.claude/` 底下的目錄結構
