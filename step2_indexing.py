# step2_indexing.py

# ── 환경변수 로딩 ──────────────────────────────────────────
from dotenv import load_dotenv  # .env 파일을 읽어주는 라이브러리
import os                        # 환경변수 꺼내쓸 때 필요

# ── LangChain 관련 ────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
# PDF 파일을 읽어서 텍스트로 변환해주는 로더
# 페이지별로 문서 객체(Document)를 만들어줌

from langchain_text_splitters import RecursiveCharacterTextSplitter
# 긴 텍스트를 일정 크기로 잘라주는 도구 (청킹)
# Recursive = 문단 → 문장 → 단어 순으로 자연스러운 위치에서 자름

from langchain_openai import OpenAIEmbeddings
# 텍스트를 숫자 벡터로 변환해주는 임베딩 모델
# GMS를 통해 OpenAI의 text-embedding-3-small 사용

from langchain_community.vectorstores import FAISS
# 벡터들을 저장하고 유사도 검색해주는 저장소
# Facebook이 만든 라이브러리, 로컬에서 무료로 사용 가능

load_dotenv()  # .env 파일의 API_KEY, API_BASE를 메모리에 올림

# 환경변수 확실하게 고정 (LangChain 내부에서 덮어씌워지는 경우 방지)
os.environ["OPENAI_API_BASE"] = "https://gms.ssafy.io/gmsapi/api.openai.com/v1"

# ── 1. PDF 로딩 ──────────────────────────────────────────
print("1. PDF 로딩 중...")
loader = PyPDFLoader("data/2024 알기 쉬운 의료급여제도.pdf")
# PyPDFLoader 객체 생성 (아직 읽기 전, 설정만 함)

docs = loader.load()
# 실제로 PDF를 읽어서 페이지별 Document 객체 리스트로 반환
# docs[0] = 1페이지, docs[1] = 2페이지 ...
# 각 Document는 page_content(텍스트)와 metadata(페이지번호 등)를 가짐

print(f"   총 페이지 수: {len(docs)}")

# ── 2. 청킹 ──────────────────────────────────────────────
print("2. 청킹 중...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,    # 청크 하나의 최대 글자 수
                       # 300자면 약 2~3문장 분량
    chunk_overlap=50   # 앞 청크와 뒷 청크가 50자 겹침
                       # 겹치는 이유: 문장이 청크 경계에서 잘렸을 때
                       # 앞뒤 문맥을 이어주기 위해
)

# chunk_overlap 예시:
# 청크1: "1종 수급권자의 외래 본인부담률은 1차 의원"
# 청크2: "의원 기준으로 1,000원이며 2차 병원은..."
#         ↑ "의원" 이 겹쳐서 문맥이 이어짐

split_docs = text_splitter.split_documents(docs)
# docs(페이지 단위)를 chunk_size 기준으로 잘게 분리
# 결과: 300자짜리 청크들의 리스트

print(f"   총 청크 수: {len(split_docs)}")

# 청크 샘플 확인 (표 데이터가 어떻게 나오는지 육안 확인용)
print("\n===== 청크 샘플 3개 =====")
for i in range(3):
    doc = split_docs[i]
    print(f"\n[Chunk {i+1}] page={doc.metadata.get('page')}")
    print(doc.page_content[:500].replace("\n", " "))

# ── 3. 임베딩 모델 ───────────────────────────────────────
print("\n3. 임베딩 모델 설정 중...")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",              # 사용할 임베딩 모델명
    openai_api_key=os.getenv("OPENAI_API_KEY"),  # .env에서 GMS 키 꺼냄
    openai_api_base=os.getenv("OPENAI_API_BASE"), # GMS 엔드포인트로 교체
    check_embedding_ctx_length=False,            # GMS는 토큰 배열 대신 텍스트 문자열만 지원
)
# 아직 API 호출 안 함! 설정 객체만 만든 상태

# ── 4. FAISS 저장 ─────────────────────────────────────────
print("4. FAISS 저장 중... (시간 걸릴 수 있어요)")
vectorstore = FAISS.from_documents(split_docs, embedding_model)
# 여기서 실제로 두 가지가 동시에 일어남:
#   1) 각 청크를 embedding_model로 벡터화 (GMS API 호출)
#   2) 벡터들을 FAISS 인덱스에 저장
# 청크 수가 많을수록 시간이 오래 걸림

vectorstore.save_local("faiss_index")
# FAISS 인덱스를 로컬 폴더에 저장
# faiss_index/ 폴더가 생기고 안에 2개 파일이 만들어짐:
#   - index.faiss : 벡터 데이터
#   - index.pkl   : 청크 텍스트 + 메타데이터

print("   저장 완료! → faiss_index/ 폴더 생성됨")