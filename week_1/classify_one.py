import json
from dotenv import load_dotenv
from openai import OpenAI
import os

# --- 1. 데이터 불러오기 (보일러플레이트 - 복붙 OK) ---
with open("dataset.jsonl", "r", encoding="utf-8") as f:
    tickets = [json.loads(line) for line in f]

# --- 2. 클라이언트 셋업 (보일러플레이트 - 복붙 OK) ---
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1"
)

# ⚠️ [직접 작성 추천] 분류 규칙을 담은 system 프롬프트
# 과제의 핵심 학습 포인트입니다. 지금은 제가 예시로 채워드리지만,
# v1->v2 개선 단계에서는 이 부분을 본인이 직접 고치셔야 해요.
# 지금 단계에서도 한번 직접 스키마 정의를 과제 md 보면서 본인 말로 써보시는 걸 추천드려요.
SYSTEM_PROMPT = """당신은 고객 문의 티켓을 분류하는 분류기입니다.
다음 JSON 스키마로만 응답하세요. 다른 설명이나 텍스트는 절대 포함하지 마세요.

{
  "intent": "order_change | shipping_issue | payment_issue | refund_exchange | other",
  "urgency": "low | medium | high",
  "needs_clarification": true 또는 false,
  "route_to": "order_ops | shipping_ops | billing_ops | returns_ops | human_support"
}

분류 기준:
- intent: order_change(주문 수정/취소/주소 변경), shipping_issue(배송 지연/누락), payment_issue(결제 실패/중복 결제), refund_exchange(반품/환불/교환), other(애매한 경우)
- urgency: high(결제 이상, 분실/오배송, 불만 고조), medium(처리 필요하지만 긴급 아님), low(일반 문의)
- needs_clarification: 텍스트만으로 판단하기 어려우면 true
- route_to: intent에 대응하는 담당 부서 (order_ops, shipping_ops, billing_ops, returns_ops, human_support)
"""

# --- 3. 첫 번째 티켓 1건만 테스트 (보일러플레이트 - 복붙 OK) ---
ticket = tickets[0]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ticket["customer_message"]}
    ]
)

print("--- 입력 ---")
print(ticket["customer_message"])
print("--- 모델 응답 ---")
print(response.choices[0].message.content)
print("--- 정답(expected_output) ---")
print(ticket["expected_output"])

print()

# ⚠️ [직접 작성 추천]
# response.choices[0].message.content를 json.loads()로 변환해서
# predicted라는 변수에 dict로 담아보세요.
# 그다음 predicted == ticket["expected_output"] 으로 직접 비교해서
# True/False가 나오는지 확인해보세요.

predict = json.loads(response.choices[0].message.content)
print(predict)

is_corecttd = ticket["expected_output"] == predict
print(f'맞은 여부: {is_corecttd}')