import os, requests
from pathlib import Path

MODEL = "meta-llama/llama-3.2-1b-instruct"
URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = Path(__file__).resolve().parents[1] / "data" / "llm_response.txt"
PROMPT = "Сформулируй одно предложение о важности code review."

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    raise SystemExit("OPENROUTER_API_KEY is missing; export it or put into .env")
resp = requests.post(
    URL,
    json={"model": MODEL, "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": PROMPT},
    ]},
    headers={
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/elenagernichenko/github-analyzer",
        "X-Title": "github-analyzer",
    },
    timeout=60,
)
resp.raise_for_status()
text = resp.json()["choices"][0]["message"]["content"].strip()
OUT.write_text(text)
print(f"Saved response to {OUT}")
