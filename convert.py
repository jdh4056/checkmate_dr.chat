import pandas as pd
import json

# CSV 경로 설정
csv_path = 'Disease_symptom_and_patient_profile_dataset.csv'

# CSV 불러오기
df = pd.read_csv(csv_path)

# 증상 컬럼
symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']

# 결과 리스트
json_data = []

# 변환 수행
for _, row in df.iterrows():
    disease = row['Disease']
    age = row['Age']
    gender = row['Gender']
    bp = row['Blood Pressure']
    chol = row['Cholesterol Level']
    outcome = row['Outcome Variable']
    
    # 'Yes'인 증상 추출
    symptoms = [s for s in symptom_cols if row[s] == 'Yes']
    if not symptoms:
        continue
    
    # 자연어 문장 생성
    symptom_str = ', '.join(symptoms)
    text = f"{symptom_str} 증상이 있는 경우 {disease}일 수 있습니다."
    
    # JSON 구조
    json_data.append({
        "text": text,
        "metadata": {
            "disease": disease,
            "symptoms": symptoms,
            "age": int(age),
            "gender": gender,
            "blood_pressure": bp,
            "cholesterol": chol,
            "outcome": outcome
        }
    })

# 저장
with open('disease_rag_with_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print("✅ JSON 저장 완료")
