# test2.py
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# 추가
os.environ["OPENAI_API_BASE"] = "https://gms.ssafy.io/gmsapi/api.openai.com/v1"

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

# 청크 여러개 한번에 임베딩 테스트
# (step2에서 실패한 embed_documents 방식)
texts = ["테스트 문장 1입니다.", "테스트 문장 2입니다."]
result = embedding_model.embed_documents(texts)
print(f"성공! 벡터 수: {len(result)}")
print(f"벡터 차원: {len(result[0])}")

