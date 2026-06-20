import json

# dataset.jsonl이 같은 week_1 폴더 안에 있다고 가정
with open("dataset.jsonl", "r", encoding="utf-8") as f:
    tickets = []
    for line in f:
        ticket = json.loads(line)
        tickets.append(ticket)

# 잘 불러와졌는지 확인
print(f"불러온 티켓 개수: {len(tickets)}")
print("--- 첫 번째 티켓 ---")
print(tickets[0])