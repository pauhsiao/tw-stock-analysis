#!/usr/bin/env python3
"""台股每日分析 — FinMind + Gemini + Pushover"""

import os
import re
import sys
import json
import time
import tempfile
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 設定 ────────────────────────────────────────────────────────────────────
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")
PUSHOVER_API_TOKEN = os.getenv("PUSHOVER_API_TOKEN", "")
STOCK_LIST = [s.strip() for s in os.getenv("STOCK_LIST", "2330").split(",") if s.strip()]

FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models"
    f"/{GEMINI_MODEL}:generateContent"
)

# ─── FinMind 資料抓取 ─────────────────────────────────────────────────────────

def _finmind_get(dataset: str, data_id: str, start_date: str = "", end_date: str = "") -> list:
    params = {"dataset": dataset, "data_id": data_id, "token": FINMIND_TOKEN}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    try:
        resp = requests.get(FINMIND_API, params=params, timeout=30)
        data = resp.json()
        if data.get("status") == 200:
            return data.get("data", [])
        logger.warning(f"FinMind {dataset} {data_id}: {data.get('msg', 'error')}")
    except Exception as e:
        logger.error(f"FinMind 請求失敗 {dataset} {data_id}: {e}")
    return []


def fetch_price(stock_id: str, days: int = 80) -> pd.DataFrame:
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    # Try TWSE first, then OTC (上櫃)
    rows = _finmind_get("TaiwanStockPrice", stock_id, start_date, end_date)
    if not rows:
        rows = _finmind_get("TaiwanStockPrice", stock_id, start_date, end_date)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "max", "min", "Trading_Volume", "Trading_money"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)
    return df


def fetch_info(stock_id: str) -> dict:
    rows = _finmind_get("TaiwanStockInfo", stock_id)
    return rows[0] if rows else {"stock_name": stock_id, "industry_category": "未知"}


def fetch_news(stock_id: str, days: int = 3) -> list:
    # FinMind TaiwanStockNews: omit end_date, use start_date only
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return _finmind_get("TaiwanStockNews", stock_id, start_date)[-8:]


def fetch_institutional(stock_id: str) -> dict:
    """三大法人買賣超（最新一日）"""
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    rows = _finmind_get("TaiwanStockInstitutionalInvestors", stock_id, start_date)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    latest = df[df["date"] == df["date"].max()]
    result = {}
    for _, row in latest.iterrows():
        name = str(row.get("name", ""))
        try:
            buy_sell = float(row.get("buy", 0) or 0) - float(row.get("sell", 0) or 0)
        except (ValueError, TypeError):
            buy_sell = 0
        if "外資" in name:
            result["foreign"] = buy_sell
        elif "投信" in name:
            result["trust"] = buy_sell
        elif "自營" in name:
            result["dealer"] = buy_sell
    return result

# ─── 技術指標 ─────────────────────────────────────────────────────────────────

def calc_indicators(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {}
    close = df["close"]
    high = df["max"]
    low = df["min"]
    vol = df["Trading_Volume"]

    ind: dict = {}
    ind["date"] = df["date"].iloc[-1].strftime("%Y/%m/%d")
    ind["close"] = float(close.iloc[-1])
    ind["open"] = float(df["open"].iloc[-1])
    ind["high"] = float(high.iloc[-1])
    ind["low"] = float(low.iloc[-1])
    ind["volume"] = float(vol.iloc[-1])

    if len(df) >= 2:
        prev = float(close.iloc[-2])
        ind["prev_close"] = prev
        ind["change"] = ind["close"] - prev
        ind["change_pct"] = ind["change"] / prev * 100

    for n in [5, 10, 20, 60]:
        if len(df) >= n:
            ind[f"ma{n}"] = float(close.rolling(n).mean().iloc[-1])

    if all(f"ma{n}" in ind for n in [5, 10, 20]):
        ind["bullish"] = ind["ma5"] > ind["ma10"] > ind["ma20"]

    if len(df) >= 9:
        hr = high.rolling(9).max()
        lr = low.rolling(9).min()
        denom = hr - lr
        rsv = ((close - lr) / denom * 100).where(denom > 0, 50)
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        ind["k"] = float(k.iloc[-1])
        ind["d"] = float(d.iloc[-1])

    if len(df) >= 15:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("inf"))
        ind["rsi"] = float((100 - 100 / (1 + rs)).iloc[-1])

    if len(df) >= 5:
        vm5 = float(vol.rolling(5).mean().iloc[-1])
        if vm5 > 0:
            ind["vol_ratio"] = ind["volume"] / vm5

    if len(df) >= 20:
        ind["support20"] = float(low.rolling(20).min().iloc[-1])
        ind["resist20"] = float(high.rolling(20).max().iloc[-1])

    return ind

# ─── LLM 分析 ────────────────────────────────────────────────────────────────

def _fmt(v, fmt=".2f", fallback="N/A"):
    if isinstance(v, float) and not (v != v):  # not NaN
        return format(v, fmt)
    return fallback


def analyze(stock_id: str, name: str, industry: str,
            ind: dict, news: list, inst: dict) -> str:
    ma_parts = [f"MA{n}={_fmt(ind.get(f'ma{n}'))}" for n in [5, 10, 20, 60] if f"ma{n}" in ind]
    ma_str = " | ".join(ma_parts) or "N/A"

    news_str = "\n".join(
        f"- {n.get('title', '')} ({n.get('date', '')})" for n in news
    ) or "近期無新聞資料"

    inst_str = (
        f"外資：{inst.get('foreign', 0):+,.0f}張  "
        f"投信：{inst.get('trust', 0):+,.0f}張  "
        f"自營：{inst.get('dealer', 0):+,.0f}張"
        if inst else "三大法人資料無"
    )

    prompt = f"""你是專業台股分析師，用繁體中文分析以下個股，給具體操作建議。

股票：{stock_id}（{name}）| 產業：{industry} | 日期：{ind.get('date', '今日')}

行情：收盤 {_fmt(ind.get('close'))} | 漲跌 {_fmt(ind.get('change_pct'), '+.2f')}%
     開 {_fmt(ind.get('open'))} 高 {_fmt(ind.get('high'))} 低 {_fmt(ind.get('low'))}

均線：{ma_str}
多頭排列：{"是 ✅" if ind.get('bullish') else "否 ❌"}
KD：K={_fmt(ind.get('k'), '.1f')} D={_fmt(ind.get('d'), '.1f')}
RSI(14)：{_fmt(ind.get('rsi'), '.1f')}
量比：{_fmt(ind.get('vol_ratio'), '.2f')}x
支撐：{_fmt(ind.get('support20'))} | 壓力：{_fmt(ind.get('resist20'))}

三大法人（最新）：{inst_str}

近期新聞：
{news_str}

請回答（格式精簡，不要廢話）：
**1. 多空判斷** — 一句話說明現在偏多/偏空/盤整
**2. 核心觀點** — 2-3個最重要的技術/籌碼觀察
**3. 操作建議** — 明確說買入/持有/賣出 + 簡短理由
**4. 關鍵價位** — 買點 / 止損 / 目標價（用數字，若無法判斷說明原因）
**5. 主要風險** — 最重要的1-2個風險點"""

    for attempt in range(3):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800},
                },
                timeout=60,
            )
            result = resp.json()
            if "candidates" in result:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            if result.get("error", {}).get("code") == 429:
                wait = 30 * (attempt + 1)
                logger.warning(f"{stock_id} Gemini 429，等 {wait}s 後重試")
                time.sleep(wait)
                continue
            logger.error(f"{stock_id} Gemini 異常: {json.dumps(result)[:200]}")
            break
        except Exception as e:
            logger.error(f"{stock_id} Gemini 失敗: {e}")
            break
    return "⚠️ LLM 分析失敗"

# ─── 報告 HTML ───────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'PingFang TC',
    'Microsoft JhengHei', 'Noto Sans TC', sans-serif;
  font-size: 13px; line-height: 1.6; color: #1a1a1a;
  background: #fff; padding: 18px 20px; max-width: 780px;
}
.report-title {
  font-size: 18px; font-weight: 700; color: #0a0a23;
  border-bottom: 3px solid #0070f3; padding-bottom: 8px; margin-bottom: 18px;
}
.stock-section { margin-bottom: 18px; }
.stock-head {
  font-size: 14px; font-weight: 700; color: #fff;
  background: #0070f3; padding: 5px 12px; border-radius: 5px; margin-bottom: 8px;
}
.price-row {
  background: #f0f7ff; border: 1px solid #c8e0ff;
  border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;
}
.price-main { font-size: 22px; font-weight: 800; color: #0a0a23; }
.up { color: #c0392b; } .down { color: #1a7a4a; } .flat { color: #888; }
.metrics { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 6px; }
.metric .lbl { font-size: 11px; color: #888; }
.metric .val { font-size: 13px; font-weight: 600; }
.analysis { font-size: 12.5px; line-height: 1.65; }
.analysis strong { color: #0070f3; }
.analysis p { margin: 3px 0; }
.analysis ul, .analysis ol { margin: 4px 0 4px 16px; }
.analysis li { margin: 2px 0; }
hr { border: none; border-top: 1px solid #e5e5e5; margin: 14px 0; }
"""

def _md_to_html(text: str) -> str:
    try:
        import markdown2
        return markdown2.markdown(text, extras=["fenced-code-blocks", "break-on-newline"])
    except Exception:
        # Fallback: basic conversion
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'^#{1,3} (.+)$', r'<strong>\1</strong>', text, flags=re.M)
        lines = text.split('\n')
        out, in_ul = [], False
        for line in lines:
            if re.match(r'^[•\-\*] ', line):
                if not in_ul:
                    out.append('<ul>')
                    in_ul = True
                out.append(f'<li>{line[2:]}</li>')
            else:
                if in_ul:
                    out.append('</ul>')
                    in_ul = False
                out.append(f'<p>{line}</p>' if line.strip() else '')
        if in_ul:
            out.append('</ul>')
        return '\n'.join(out)


def build_html(results: list, date_str: str) -> str:
    parts = [
        f"<html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>",
        f"<div class='report-title'>📈 台股每日分析 — {date_str}</div>",
    ]

    for r in results:
        ind = r["indicators"]
        cp = ind.get("change_pct", 0)
        cls = "up" if cp > 0 else "down" if cp < 0 else "flat"
        sign = "+" if cp > 0 else ""

        def m(key, fmt=".2f"):
            return _fmt(ind.get(key), fmt)

        parts.append(f"""
<div class='stock-section'>
  <div class='stock-head'>{r['stock_id']} {r['stock_name']} | {r['industry']}</div>
  <div class='price-row'>
    <span class='price-main'>{m('close')}</span>
    <span class='{cls}'> {sign}{m('change_pct')}%（{sign}{m('change')}）</span>
    <div class='metrics'>
      <div class='metric'><div class='lbl'>開</div><div class='val'>{m('open')}</div></div>
      <div class='metric'><div class='lbl'>高</div><div class='val'>{m('high')}</div></div>
      <div class='metric'><div class='lbl'>低</div><div class='val'>{m('low')}</div></div>
      <div class='metric'><div class='lbl'>MA5</div><div class='val'>{m('ma5')}</div></div>
      <div class='metric'><div class='lbl'>MA20</div><div class='val'>{m('ma20')}</div></div>
      <div class='metric'><div class='lbl'>RSI</div><div class='val'>{m('rsi', '.1f')}</div></div>
      <div class='metric'><div class='lbl'>K/D</div><div class='val'>{m('k', '.1f')}/{m('d', '.1f')}</div></div>
      <div class='metric'><div class='lbl'>量比</div><div class='val'>{m('vol_ratio', '.2f')}x</div></div>
    </div>
  </div>
  <div class='analysis'>{_md_to_html(r['analysis'])}</div>
</div>
<hr>""")

    parts.append("</body></html>")
    return "\n".join(parts)

# ─── 圖片 & Pushover ──────────────────────────────────────────────────────────

def html_to_image(html: str) -> Optional[str]:
    try:
        import imgkit
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        imgkit.from_string(html, tmp.name, options={
            "format": "jpg", "quality": "75", "width": "780",
            "quiet": "", "encoding": "UTF-8", "disable-smart-width": "",
        })
        size = os.path.getsize(tmp.name)
        logger.info(f"報告圖片：{size // 1024} KB")
        return tmp.name
    except Exception as e:
        logger.error(f"圖片生成失敗: {e}")
        return None


def send_pushover(image_path: Optional[str], title: str, message: str):
    if not (PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN):
        logger.warning("Pushover 未設定，略過")
        return
    url = "https://api.pushover.net/1/messages.json"
    payload = {"token": PUSHOVER_API_TOKEN, "user": PUSHOVER_USER_KEY,
               "title": title, "message": message}
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                r = requests.post(url, data=payload,
                                  files={"attachment": ("report.jpg", f, "image/jpeg")},
                                  timeout=60)
        else:
            r = requests.post(url, data=payload, timeout=30)
        if r.status_code == 200 and r.json().get("status") == 1:
            logger.info("Pushover 發送成功 ✅")
        else:
            logger.error(f"Pushover 失敗: {r.text}")
    except Exception as e:
        logger.error(f"Pushover 例外: {e}")
    finally:
        if image_path:
            try:
                os.unlink(image_path)
            except Exception:
                pass

# ─── 主程式 ──────────────────────────────────────────────────────────────────

def main():
    date_str = datetime.now().strftime("%Y/%m/%d")
    logger.info(f"台股分析開始 {date_str}，共 {len(STOCK_LIST)} 支：{', '.join(STOCK_LIST)}")

    if not FINMIND_TOKEN:
        logger.error("❌ 缺少 FINMIND_TOKEN")
        sys.exit(1)

    results = []
    for i, stock_id in enumerate(STOCK_LIST):
        logger.info(f"── {stock_id} ──")
        info = fetch_info(stock_id)
        df = fetch_price(stock_id)
        if df.empty:
            logger.warning(f"{stock_id}: 無行情資料，跳過")
            continue
        ind = calc_indicators(df)
        news = fetch_news(stock_id)
        inst = fetch_institutional(stock_id)
        # Space out Gemini calls: free tier = 15 RPM, wait 5s between stocks
        if i > 0 and GEMINI_API_KEY:
            time.sleep(5)
        llm_result = analyze(
            stock_id,
            info.get("stock_name", stock_id),
            info.get("industry_category", "未知"),
            ind, news, inst,
        ) if GEMINI_API_KEY else "（未設定 Gemini API Key）"

        results.append({
            "stock_id": stock_id,
            "stock_name": info.get("stock_name", stock_id),
            "industry": info.get("industry_category", "未知"),
            "indicators": ind,
            "analysis": llm_result,
        })

    if not results:
        logger.error("所有股票均無資料")
        send_pushover(None, f"❌ 台股分析失敗 {date_str}", "所有股票資料取得失敗，請確認 FinMind Token 和股票代碼")
        sys.exit(1)

    # 生成摘要
    summary_lines = []
    for r in results:
        cp = r["indicators"].get("change_pct", 0)
        sign = "+" if cp > 0 else ""
        summary_lines.append(f"{r['stock_id']} {r['stock_name']}: {sign}{cp:.2f}%")
    summary = "\n".join(summary_lines)

    # 生成報告圖片
    html = build_html(results, date_str)
    image_path = html_to_image(html)

    # 發送 Pushover
    send_pushover(image_path, f"📈 台股分析 {date_str}", summary)
    logger.info("完成！")


if __name__ == "__main__":
    main()
