from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings

load_dotenv()  # .env 파일 읽기

## 임베딩 모델 객체 생성하기
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

## 임베딩 모델을 통해서 예시 쿼리를 임베딩 벡터화 시키기
result = embeddings.embed_query("테스트 문장입니다.")
print(f"임베딩 벡터 차원: {len(result)}")  # 1536 나오면 성공
print("GMS 연결 성공!")

# print(f"임베딩 벡터 출력: {result}")  # 벡터가 나오네요
