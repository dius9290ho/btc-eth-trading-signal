"""Hourly BTC/ETH signal monitor; runs independently of Streamlit sessions."""
import json
import os
import sys
import base64
import requests
from datetime import datetime, timezone
from pathlib import Path

from app import daily_candles, intraday_monitor

REPO = os.environ.get("GITHUB_REPOSITORY", "")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
BRANCH = os.environ.get("GITHUB_REF_NAME", "main")
STATE_PATH = "monitor_state.json"
MARKETS = ("KRW-BTC", "KRW-ETH")
STRATEGY = os.environ.get("SIGNAL_STRATEGY", "v14")


def github_state():
    if not REPO or not TOKEN:
        return {}, None
    url = f"https://api.github.com/repos/{REPO}/contents/{STATE_PATH}"
    r = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}, params={"ref": BRANCH}, timeout=20)
    if r.status_code == 404:
        return {}, None
    r.raise_for_status()
    payload = r.json()
    return json.loads(base64.b64decode(payload["content"]).decode("utf-8")), payload["sha"]


def save_state(state, sha):
    url = f"https://api.github.com/repos/{REPO}/contents/{STATE_PATH}"
    body = {"message": "Update hourly signal monitoring state", "branch": BRANCH,
            "content": base64.b64encode((json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode()).decode()}
    if sha:
        body["sha"] = sha
    r = requests.put(url, json=body, headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}, timeout=25)
    r.raise_for_status()


def telegram(message):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram not configured; alert not delivered.", file=sys.stderr)
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": message}, timeout=20)
    r.raise_for_status()
    if not r.json().get("ok"):
        raise RuntimeError("Telegram API did not confirm delivery")
    return True


def main():
    if not REPO or not TOKEN:
        raise RuntimeError("GitHub Actions token and repository are required for persistent state")
    previous, sha = github_state()
    updated = dict(previous)
    failed = False
    for market in MARKETS:
        key = f"{market}:{STRATEGY}"
        try:
            result = intraday_monitor(daily_candles(market, count=250), market,
                                      strategy=STRATEGY, previous=previous.get(key))
            current = result["state"]
            if result["alert_event"]:
                message = (f"🔔 {market} 매매신호 변경\n"
                           f"{result['previous_label']} → {result['signal_label']}\n"
                           f"가격: ₩{current['price']:,.0f}\n"
                           f"판단: {result['reason']}\n"
                           f"시간: {current['checked_at']}\n"
                           "※ 일봉 장중 미확정 신호이며 투자 권유가 아닙니다.")
                if not telegram(message):
                    # Do not acknowledge a signal change until the notification can be sent.
                    failed = True
                    continue
            updated[key] = current
            print(f"{market}: {current['signal_label']} alert={result['alert_event']}")
        except Exception as exc:
            failed = True
            print(f"{market}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if updated != previous:
        save_state(updated, sha)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
