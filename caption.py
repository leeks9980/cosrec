# -*- coding: utf-8 -*-

import google.generativeai as genai
import pandas as pd
import sqlite3
import time
import os

# 0. Gemini API 키 설정
os.environ["GOOGLE_API_KEY"] = "AIzaSyBUZ1IrgpVvYYNeZGgYIs9JCOX3Zyf4AO8"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 1. CSV 불러오기 및 성분 리스트 추출
csv_path = r"C:\Users\user\Desktop\Github\cosrec\output1.csv"
df = pd.read_csv(csv_path)

ingredient_set = set()
for col in df.columns[3:]:  # 첫 열은 제품명으로 가정
    ingredient_set.update(df[col].dropna().astype(str).tolist())

ingredient_list = sorted(ingredient_set)

# 2. Gemini 모델 초기화
model = genai.GenerativeModel('gemini-2.0-flash')

# 3. 필수 항목 리스트 (형식 검증용)
required_fields = [
    "효과:", "부작용:", "주의사항:", "적합한 피부 타입:",
    "권장 농도:", "함께 사용하면 안 되는 성분:", "포함된 대표 제품:", "사용 시기/계절:"
]

# 4. 설명 생성 함수 (프롬프트 강화 + Gemini API 활용)
def generate_description(ingredient):
    prompt = (
        f"당신은 피부 성분 분석 전문가입니다.\n\n"
        f"다음 성분에 대한 정보를 아래 양식에 맞춰 정확하고 간결하게 작성해주세요.\n\n"
        f"성분: {ingredient}\n\n"
        f"[출력 양식]\n"
        f"- 효과:\n"
        f"- 부작용:\n"
        f"- 주의사항:\n"
        f"- 적합한 피부 타입:\n"
        f"- 권장 농도:\n"
        f"- 함께 사용하면 안 되는 성분:\n"
        f"- 포함된 대표 제품:\n"
        f"- 사용 시기/계절:\n\n"
        f"※ 위 항목은 반드시 모두 포함하며, 각 항목의 제목과 형식을 그대로 유지해주세요.\n"
        f"※ 아래 표현은 절대 사용하지 마세요: '치료', '개선', '질병명', '가장 좋다', '탁월하다', '비교'\n"
    )

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()

        # 검증
        if all(field in content for field in required_fields):
            return content
        else:
            return "[형식 오류] 일부 항목이 누락되었거나 잘못된 형식입니다.\n\n" + content
    except Exception as e:
        return f"[ERROR] {str(e)}"

# 5. 성분별 설명 생성 및 결과 저장
results = []
for i, ingredient in enumerate(ingredient_list):
    print(f"[{i+1}/{len(ingredient_list)}] Generating info for: {ingredient}")
    description = generate_description(ingredient)
    results.append({
        "성분명": ingredient,
        "설명": description
    })
    time.sleep(1.5)  # Gemini API 과도한 호출 방지

# 6. CSV 및 SQLite 저장
result_df = pd.DataFrame(results)
result_df.to_csv("효과DB_성분설명_gemini.csv", index=False, encoding="utf-8-sig")

conn = sqlite3.connect("skincare_ingredient.db")
result_df.to_sql("ingredient_descriptions", conn, if_exists="replace", index=False)
conn.close()

print("✅ Gemini 기반 CSV 및 SQLite DB 저장 완료")
