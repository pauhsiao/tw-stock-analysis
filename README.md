# 台股每日分析

每日自動分析台股，使用 FinMind 官方資料 + Gemini AI 分析，透過 Pushover 推播報告圖片。

## 設定

### GitHub Secrets（必填）
| Secret | 說明 |
|--------|------|
| FINMIND_TOKEN | [FinMind](https://finmindtrade.com/) API Token |
| GEMINI_API_KEY | Google Gemini API Key |
| PUSHOVER_USER_KEY | Pushover 用戶 Key |
| PUSHOVER_API_TOKEN | Pushover 應用 Token |

### GitHub Variables（可選）
| Variable | 預設值 | 說明 |
|----------|--------|------|
| STOCK_LIST | 2330,2327,... | 台股代碼，逗號分隔 |
| GEMINI_MODEL | gemini-2.0-flash | Gemini 模型 |

## 資料來源
- 行情、技術指標、新聞、三大法人：[FinMind](https://finmindtrade.com/)（台灣官方 TWSE 資料）
- AI 分析：Google Gemini
