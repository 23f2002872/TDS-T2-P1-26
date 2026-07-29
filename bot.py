import json
import os
import time
import traceback

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
LOG_URL = os.environ["LOG_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
LOG_FILE = os.environ.get("LOG_FILE", "run.jsonl")

client = OpenAI(base_url="https://aipipe.org/openai/v1", api_key=AIPIPE_TOKEN)

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


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running... (Ctrl+C to stop)")
    app.run_polling()


if __name__ == "__main__":
    main()
