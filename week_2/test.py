import json
from dotenv import load_dotenv
from openai import OpenAI
import os

from pydantic import BaseModel

# --- 3. JSON 로드 및 클라이언트 호출  ---
with open("data/dataset.jsonl", "r", encoding="utf-8") as f:
    datasets = []
    for line in f:
        question = json.loads(line)
        datasets.append(question)

# answer_key 로드 (with 블록 완전히 분리)
with open("data/answer_key.jsonl", encoding="utf-8") as f:
    answer_key_list = []
    for line in f:
        answer_key_list.append(json.loads(line))

print(datasets[0])
print(answer_key_list[0])