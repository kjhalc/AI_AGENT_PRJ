import json
from dotenv import load_dotenv
from google import genai
import os
import time

from pydantic import BaseModel

# --- 1. 클라이언트 셋업 ---
load_dotenv()
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options={"base_url": "https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com"}
)

# Pydantic 모델
class CopaymentResult(BaseModel):
    reason: str
    answer: str

# --- 2. 프롬프트 작성 ---
copayment_reference ="""
# [의료급여 본인부담률 핵심 참조 가이드]

## 1. 수급권자 및 기관 정의
- 1종 수급권자: 근로무능력가구, 희귀/중증난치질환자, 시설수급자 등
- 2종 수급권자: 기초생활수급자 중 1종 기준에 해당하지 않는 자
- 의료기관 종별: 1차(의원), 2차(병원/종합병원), 3차(상급종합병원)[cite: 1]

## 2. 기본 본인부담률 (일반 진료)
| 구분 | 1차(의원) | 2차(병원) | 3차(상급) | 약국 | CT/MRI/PET |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1종 입원** | 무료 | 무료 | 무료 | - | 무료 |
| **1종 외래** | 1,000원 | 1,500원 | 2,000원 | 500원 | 5% |
| **2종 입원** | 10% | 10% | 10% | - | 10% |
| **2종 외래** | 1,000원 | 15% | 15% | 500원 | 15% |
* 선택의료급여기관 미신청자: 입원 20%, 외래/약국 30% 적용[cite: 1]

## 3. 연령별 특례 (최우선 적용)
- **1세 미만 (0세):** 입원/외래 모두 무료. 단, 2·3차 기관 외래진료 이거나 특수검사(CT/MRI/PET)는 5%[cite: 1]
- **1세 미만 만성질환자:** 2차 기관 외래 진료 시 본인부담 무료[cite: 1]
- **6세 미만:** 입원 무료[cite: 1]
- **6세 이상 ~ 15세 이하:** 입원 3%[cite: 1]
- **18세 이하:** 치아 홈메우기 입원(6세 미만 무료, 6~15세 3%, 16~18세 5%), 외래(병원급 이상 5%)[cite: 1]

## 4. 질환 및 항목별 특례
- **노인 (65세 이상):**
    - 틀니: 1종 5%, 2종 15%
    - 임플란트: 1종 10%, 2종 20%
  * ※ 본인부담 보상제/상한제 적용 제외 항목임[cite: 1]
- **정신질환 (외래):**
    - 조현병: 병원급 이상 5%
    - 조현병 외 정신질환: 병원급 이상 10%[cite: 1]
- **치매질환:** 입원 및 병원급 이상 외래 5%[cite: 1]
- **분만 및 임신부:**
    - 입원: 자연분만/제왕절개 무료, 고위험 임신부 5%
    - 외래: 임신부(유산/사산 포함) 병원급 이상 5%[cite: 1]
- **추나요법:**
    - 디스크/협착증: 1종 30%, 2종 40% (단순/복잡 공통)
    - 디스크/협착증 외: 1종/2종 모두 80% (복잡추나 기준)[cite: 1]

## 5. 특수검사 (CT, MRI, PET) 상세
- 임신부, 5세 이하 조산아/저체중아, 치매질환자: 1차 기관 5%
- 1세 미만 만성질환자: 2차 기관 5%
- 조현병 등 정신질환자: 2·3차 기관 15%[cite: 1]

## 6. 기타 원칙
- 본인부담 면제자: 18세 미만, 임산부, 희귀질환자, 1세 미만(의원급) 등[cite: 1]
- 식대 본인부담: 2종 장애인 20%, 중증질환자 5%, 6세 미만/자연분만 무료[cite: 1]
"""


system_prompt = f"""
    아래는 의료급여 본인부담률 참조 데이터와 추가 조건입니다.
    질문에 대해 정확한 본인부담률을 답하세요. 답만 간결하게 작성하세요.
    반드시 JSON 형식으로 'answer'(최종 답)와 'reason'(판단 근거)을 포함해야 합니다.
    
    <추가 조건>
    어떤 조건에도 해당되지 않을 경우 반드시 "해당되지 않음" 으로만 작성할 것
    ("적용되지 않음", "해당없음" 등 다른 표현 사용 금지)
    0%인 경우에는 무료 라고 기입할 것
    숫자 콤마 및 단위 기입할 것
    어떤 조건에도 해당되지 않을 경우 해당되지 않음 으로 작성할 것
    한 개인 경우 %만 출력할 것
    여러 개 일 경우 (내용) 0,000원, (내용) 0,000원 형태로 기입할 것

    {copayment_reference}
"""


# --- 3. JSON 로드 ---
with open("data/dataset.jsonl", "r", encoding="utf-8") as f:
    datasets = []
    for line in f:
        datasets.append(json.loads(line))

with open("data/answer_key.jsonl", encoding="utf-8") as f:
    answer_key_list = []
    for line in f:
        answer_key_list.append(json.loads(line))

def generate_with_retry(client, **kwargs):
    while True:
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:
            if "429" in str(e):
                print("한도 초과, 30초 대기 후 재시도...")
                time.sleep(30)
            else:
                raise

FEW_SHOT_IDS = ["q04", "q12", "q20"]
correct = 0
total = len(datasets) - len(FEW_SHOT_IDS)

for item, answer in zip(datasets, answer_key_list):

    if item["id"] in FEW_SHOT_IDS:
        continue

    question = item["question"]
    expected = answer["expected_answer"]

    time.sleep(5)
    response = generate_with_retry(
        client,
        model="gemini-2.5-flash",
        contents=question,
        config={
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": CopaymentResult,
        }
    )

    result = json.loads(response.text)
    model_answer = result["answer"]

    if model_answer == expected:
        correct += 1

    print(f"{item['id']}: 모델={model_answer} / 정답={expected}")

print(f"\n정답률: {correct}/{total} ({correct/total*100:.1f}%)")
