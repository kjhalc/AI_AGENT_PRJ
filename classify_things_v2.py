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
- urgency: high(결제 이상, 분실/오배송, 불만 고조, 특히 문의를 처음한 게 아닌  경우 high로 할것), medium(처리 필요하지만 긴급 아님), low(일반 문의)
- needs_clarification: 텍스트만으로 판단하기 어려우면 true
- route_to: intent에 대응하는 담당 부서 (order_ops, shipping_ops, billing_ops, returns_ops, human_support)
"""

# --- 3. 첫 번째 티켓 1건만 테스트 (보일러플레이트 - 복붙 OK) ---

index = 1
results = []

for ticket in tickets:
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ticket["customer_message"]}
    ]
    )


    predict = json.loads(response.choices[0].message.content)
    


    is_correct = predict == ticket["expected_output"]
    print('예측값')
    print(predict)
    print('출력값')
    print(ticket["expected_output"])
    print(f'{index}번은 {is_correct}')

    result_item = {
        "id" : ticket["id"],
        "predict" : predict,
        "expected" : ticket["expected_output"],
        "is_correct" : is_correct
    }

    results.append(result_item)


    index += 1

    

#전체 맞은 갯수 카운팅

count = 0

for item in results:
    if item["is_correct"]:
        count+=1

print(f'True 개수는 {count}개입니다')

with open("results_v2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)     

