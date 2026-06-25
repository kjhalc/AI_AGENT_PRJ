# %%
# 셀 1: 라이브러리 임포트 및 환경변수 설정
import json
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
os.environ["OPENAI_API_BASE"] = "https://gms.ssafy.io/gmsapi/api.openai.com/v1"
print("임포트 완료!")

# %%
# 셀 2: 임베딩 모델 설정
# Step 2와 반드시 동일한 모델 사용!
# 다른 모델 쓰면 벡터 공간이 달라져서 검색 결과가 완전히 틀려짐
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    check_embedding_ctx_length=False
)
print("임베딩 모델 설정 완료!")

# %%
# 셀 3: FAISS 인덱스 불러오기
# Step 2에서 save_local()로 저장한 폴더를 다시 메모리로 불러옴
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
    # 로컬에서 내가 만든 파일이라 안전함
)
print("FAISS 인덱스 로딩 완료!")

# %%
# 셀 4: Golden Dataset 읽기
def load_golden_dataset(path):
    # jsonl = 한 줄에 json 하나씩 들어있는 파일
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

golden_data = load_golden_dataset("golden_dataset.jsonl")
print(f"총 {len(golden_data)}문제 로딩 완료!")

# 어떤 질문들이 있는지 확인
for item in golden_data:
    print(f"  {item['id']} ({item['difficulty']}): {item['question']}")

# %%
# 셀 5: 질문 하나만 먼저 테스트 (q01)
# 전체 돌리기 전에 검색이 잘 되는지 먼저 확인
test_question = golden_data[0]["question"]
print(f"테스트 질문: {test_question}")

results = vectorstore.similarity_search(test_question, k=3)
# 질문 → 벡터 변환 → faiss에서 유사 벡터 Top-3 검색

print(f"\n── 검색된 Top-3 청크 ──")
for i, doc in enumerate(results, start=1):
    print(f"\n[Top {i}] page={doc.metadata.get('page', '?')}")
    print(doc.page_content[:300].replace("\n", " "))

# %%
# 셀 6: 5문제 전체 검색 + 결과 파일 저장
def normalize_text(text):
    # 공백/줄바꿈 통일 (PDF 추출 텍스트는 공백이 불규칙함)
    return " ".join(text.split())

OUTPUT_FILE = "retrieval_results.txt"
TOP_K = 3
success_count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for item in golden_data:
        question = item["question"]
        evidence_text = item["evidence_text"]
        difficulty = item["difficulty"]

        # 검색 실행
        retrieved_docs = vectorstore.similarity_search(question, k=TOP_K)

        # evidence_text 포함 여부 확인
        found = False
        norm_evidence = normalize_text(evidence_text)

        output = "\n" + "="*50 + "\n"
        output += f"[{item['id']}] 난이도: {difficulty}\n"
        output += f"[질문] {question}\n"
        output += f"[정답 근거] {evidence_text}\n"
        output += f"\n── Top-{TOP_K} 검색 결과 ──\n"

        for i, doc in enumerate(retrieved_docs, start=1):
            chunk_text = doc.page_content
            page = doc.metadata.get("page", "?")
            output += f"\n[Top {i}] page={page}\n"
            output += chunk_text[:300].replace("\n", " ") + "\n"

            if norm_evidence in normalize_text(chunk_text):
                found = True

        output += "\n[결과] "
        if found:
            output += "✅ 검색 성공\n"
            success_count += 1
        else:
            output += "❌ 검색 실패\n"

        print(output)
        out.write(output)

    # 최종 요약
    total = len(golden_data)
    summary = "\n" + "="*50 + "\n"
    summary += f"📊 최종 결과: {success_count}/{total} 성공\n"
    summary += f"검색 성공률: {success_count/total:.0%}\n"
    print(summary)
    out.write(summary)

print(f"결과 저장 완료 → {OUTPUT_FILE}")
# %%
