from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()  # .env를 읽어서 환경변수로 등록

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1"   # 핵심: GMS로 우회시키는 부분
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200,
    messages=[
        {"role": "system", "content": "당신은 고객 문의를 분류하는 분류기입니다."},
        {"role": "user", "content": "주문한 신발이 아직 안 왔어요"}
    ]
)

print(response.choices[0].message.content)