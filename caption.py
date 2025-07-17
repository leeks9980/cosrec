# captioning 후 성분 설명 DB 생성
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import time
import sqlite3

# 모델 로드
model_id = "beomi/KoAlpaca-Polyglot-12.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 성분 리스트 예시
ingredient_list = [
    "나이아신아마이드",
    "살리실산",
    "히알루론산",
    "센텔라아시아티카",
    "레티놀"
]

# 캡션 생성 함수
def generate_description(ingredient):
    prompt = f"""
당신은 피부 성분 분석 전문가입니다.
다음 성분에 대해 설명해주세요: {ingredient}

- 효과:
- 부작용:
- 주의사항:
- 적합한 피부 타입:
- 권장 농도:
- 함께 사용하면 안 되는 성분:
- 포함된 대표 제품:
- 사용 시기/계절:
"""
    output = pipe(prompt, max_new_tokens=300, do_sample=True, top_k=50, top_p=0.95)[0]['generated_text']
    return output.replace(prompt.strip(), "").strip()

# 결과 저장
results = []
for ingredient in ingredient_list:
    print(f"Generating info for: {ingredient}")
    description = generate_description(ingredient)
    results.append({
        "성분명": ingredient,
        "설명": description
    })
    time.sleep(1)  # 모델 응답 딜레이 대응

# 데이터프레임으로 정리
df = pd.DataFrame(results)
df.to_csv("효과DB_성분설명.csv", index=False, encoding="utf-8-sig")
print("효과 DB 저장 완료: 효과DB_성분설명.csv")

# CSV 로드
df = pd.read_csv("효과DB_성분설명.csv")

# SQLite 연결 및 저장
conn = sqlite3.connect("skincare_ingredient.db")
df.to_sql("ingredient_descriptions", conn, if_exists="replace", index=False)

print("SQLite DB에 저장 완료")
conn.close()
