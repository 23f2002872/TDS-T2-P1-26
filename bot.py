import json
import os
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
LOG_URL = os.environ["LOG_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
GITHUB_REPO = os.environ.get("GITHUB_REPO")  # "owner/repo", enables log push when set
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

client = OpenAI(base_url="https://aipipe.org/openai/v1", api_key=AIPIPE_TOKEN)

git_lock = threading.Lock()
git_ready = False

LOG_REPO_DIR = "log_repo"
LOG_FILENAME = "run.jsonl"
# When git push is configured, log inside the dedicated clone below so pushes
# and local writes touch the same file; otherwise just log next to the app.
LOG_FILE = (
    os.path.join(LOG_REPO_DIR, LOG_FILENAME)
    if (GITHUB_REPO and GITHUB_TOKEN)
    else os.environ.get("LOG_FILE", LOG_FILENAME)
)


def git_push_enabled():
    return bool(GITHUB_REPO and GITHUB_TOKEN)


def _git(*args, timeout=30):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, timeout=timeout, cwd=LOG_REPO_DIR
    )


def ensure_log_repo():
    # Do our own clone into a dedicated folder rather than assuming the app's
    # own deployed files include .git — some hosts ship a built artifact
    # without version-control metadata to the runtime container, so we can't
    # rely on that being present.
    global git_ready
    if not git_push_enabled():
        return
    remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
    try:
        if not os.path.isdir(os.path.join(LOG_REPO_DIR, ".git")):
            clone = subprocess.run(
                ["git", "clone", remote_url, LOG_REPO_DIR],
                capture_output=True, text=True, timeout=30,
            )
            if clone.returncode != 0:
                print(f"[git] clone failed: {clone.stderr.strip()[:300]}")
                return
        else:
            _git("remote", "set-url", "origin", remote_url)
            _git("fetch", "origin", "main")
            _git("reset", "--hard", "origin/main")
        _git("config", "user.email", "bot@tds-telegram-bot.local")
        _git("config", "user.name", "TDS Telegram Bot")
        git_ready = True
    except Exception as exc:
        print(f"[git] setup failed, logging locally only: {exc}")


def push_log():
    if not git_ready:
        return
    with git_lock:
        try:
            _git("add", LOG_FILENAME)
            commit = _git("commit", "-m", "Update run log")
            if commit.returncode != 0:
                if "nothing to commit" not in (commit.stdout + commit.stderr).lower():
                    print(f"[git] commit issue: {commit.stdout.strip()[:200]} {commit.stderr.strip()[:200]}")
                return
            push = _git("push", "origin", "HEAD:main")
            if push.returncode != 0:
                print(f"[git] push rejected: {push.stderr.strip()[:300]}")
        except Exception as exc:
            print(f"[git] push failed: {exc}")

# Keeps the last few messages per chat, so multi-turn questions work —
# "answer the LAST message" still needs the earlier ones for context.
conversation_history = {}

SYSTEM_PROMPT = (
    "You are a careful data analyst. The user's LAST message asks a data-analysis "
    "question and tells you exactly what JSON shape to reply with. Work out the "
    "real answer (use any public data you know, e.g. official statistics or "
    "general world knowledge, or arithmetic on numbers given in the message). "
    "Reply with ONLY that exact JSON object and absolutely nothing else — no "
    "explanation, no markdown, no code fences, just the raw JSON. Match the "
    "requested keys exactly: no extra keys, no missing keys, correct nesting."
)


def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start:end + 1])


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history[-6:],
        )
        reply_text = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply_text})
        parsed = extract_json(reply_text)
        parsed["log_url"] = LOG_URL
        final_reply = json.dumps(parsed)
    except Exception as exc:
        log_event({
            "type": "error",
            "chat_id": chat_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        final_reply = json.dumps({"error": "failed to produce an answer", "log_url": LOG_URL})

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)
    threading.Thread(target=push_log, daemon=True).start()


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass


def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()


def main():
    ensure_log_repo()
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)

    # Render's free Web Service plan requires binding $PORT and responding to
    # HTTP requests to stay up — the actual bot logic is Telegram polling, not
    # HTTP, so this just satisfies that health check on the side.
    threading.Thread(target=run_health_server, daemon=True).start()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running... (Ctrl+C to stop)")
    app.run_polling()


if __name__ == "__main__":
    main()
