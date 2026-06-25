# pdf_inspect.py (구조 파악용 임시 파일)
import pdfplumber

with pdfplumber.open("data/2024 알기 쉬운 의료급여제도.pdf") as pdf:
    print(f"총 페이지 수: {len(pdf.pages)}")
    
    for i, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        if tables:  # 표가 있는 페이지만 출력
            print(f"\n[{i+1}페이지] 표 {len(tables)}개 발견")
            for j, table in enumerate(tables):
                print(f"  표{j+1} 크기: {len(table)}행 x {len(table[0])}열")
                print(f"  첫 행: {table[0]}")  # 헤더 확인

# pdf_inspect.py 에 추가
with pdfplumber.open("data/2024 알기 쉬운 의료급여제도.pdf") as pdf:
    page = pdf.pages[5]  # 6페이지 (0부터 시작)
    tables = page.extract_tables()
    
    for j, table in enumerate(tables):
        print(f"\n[표{j+1}]")
        for row in table:
            print(row)


# pdf_inspect.py에 추가
with pdfplumber.open("data/2024 알기 쉬운 의료급여제도.pdf") as pdf:
    page = pdf.pages[5]  # 6페이지
    
    # 페이지 내 모든 텍스트 위치 확인
    words = page.extract_words()
    for word in words:
        print(word)