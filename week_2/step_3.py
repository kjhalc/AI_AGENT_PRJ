import json
from dotenv import load_dotenv
from openai import OpenAI
import os

from pydantic import BaseModel

# --- 1. 클라이언트 셋업 (보일러플레이트 - 복붙 OK) ---
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1"
)

# Pydantic 모델 (여기에)
class CoTResult(BaseModel):
    step1_type: str      # 먼저 생성됨
    step2_age: str       # 그 다음
    step3_condition: str # 그 다음
    step4_institution: str
    reason: str          # 종합
    answer: str          # 마지막에 생성됨

# --- 2. 프롬프트 작성  ---
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

# few shot examples (q04, q12, q20 사용)
FEW_SHOT_EXAMPLES = """

[질문-답변 가이드 예시]

예시 1 (특례 적용으로 인한 무료 케이스)
질문: 2종 수급권자인 임신부가 자연분만으로 입원하면 본인부담률은 얼마인가요?
답변: {
    "step1_type": "2종 수급권자",
    "step2_age": "해당 없음",
    "step3_condition": "분만 → 자연분만 특례 적용",
    "step4_institution": "해당 없음",
    "reason": "분만 및 임신부 특례 → 입원 → 자연분만 = 무료",
    "answer": "무료"
}

예시 2 (병원급 이상 표현 케이스)
질문: 2종 수급권자의 조현병 외 정신질환 환자가 외래 진료를 받으면 본인부담률은 얼마인가요?
답변: {
    "step1_type": "2종 수급권자",
    "step2_age": "해당 없음",
    "step3_condition": "조현병 외 정신질환 특례 적용",
    "step4_institution": "병원급 이상",
    "reason": "정신질환 외래진료 → 조현병 외 정신질환 → 병원급 이상 10%",
    "answer": "병원급 이상 10%"
}

예시 3 (금액 계산 케이스)
질문: 1종 수급권자가 디스크로 복잡추나를 받았고 치료비가 180,000원입니다. 본인부담금은 얼마인가요?
답변: {
    "step1_type": "1종 수급권자",
    "step2_age": "해당 없음",
    "step3_condition": "추나요법 → 디스크·협착증 → 복잡추나",
    "step4_institution": "해당 없음",
    "reason": "디스크·협착증 + 복잡추나 + 1종 = 30%. 180,000 × 0.30 = 54,000원",
    "answer": "54,000원"
}
"""

system_prompt = f"""
    당신은 의료급여 본인부담률 산정 전문가입니다.
    질문에 대답하기 전에, 반드시 다음 4단계 과정을 거쳐서 사고하세요:

    1단계: 수급권자가 1종인지 2종인지 파악한다.
    2단계: 연령 특례(1세 미만, 6세 미만, 15세 이하 등)가 있는지 확인한다.
    3단계: "질환/시술 특례 여부 (치매, 분만, 정신질환, 추나요법 등)" 을 판단한다
    4단계: 특정 질환(조현병, 치매 등)이나 특정 시술(틀니, 임플란트, 추나 등)에 대한 예외 규정이 있는지 확인한다.
    5단계: 방문한 의료기관(1차, 2차, 3차)에 따른 차등이 있는지 확인한다.
    답변이 두개일 시 틀니 -원, 임플란트 -원 같이 간결하게 작성한다.
    적용되지 않을때는 "해당되지 않음" 으로 답변한다.

    최종 답변(answer)은 이전과 같이 불필요한 수식어 없이 '결과값'만 적으세요.



    {copayment_reference}

    {FEW_SHOT_EXAMPLES}
"""


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

FEW_SHOT_IDS = ["q04", "q12", "q20"]
correct = 0
total = len(datasets) - len(FEW_SHOT_IDS)

for item, answer in zip(datasets, answer_key_list):

    if item["id"] in FEW_SHOT_IDS:
        continue

    question = item["question"]
    expected = answer["expected_answer"]

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},  # 매 iteration마다 다른 질문
        ],
        temperature=0,
        response_format=CoTResult,  # 이 줄 추가
    )

    # 모델 응답
    model_answer = response.choices[0].message.parsed.answer  
    
    if model_answer == expected:
        correct += 1
    
    print(f"{item['id']}: 모델={model_answer} / 정답={expected}")  # 진행 확인용

# 루프 끝난 뒤 추가
print(f"\n정답률: {correct}/{total} ({correct/total*100:.1f}%)")


