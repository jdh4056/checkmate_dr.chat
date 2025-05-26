import pandas as pd
import json

# CSV 경로 설정
csv_path = 'Disease_symptom_and_patient_profile_dataset.csv' #

# CSV 불러오기
df = pd.read_csv(csv_path)

# 증상 컬럼
# 참고: 만약 '콧물'과 같은 새로운 증상 컬럼이 CSV에 있다면 여기에 추가하세요.
# 예: symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Nasal Discharge']
symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing'] #

# 결과 리스트
json_data = []

# 변환 수행
for _, row in df.iterrows():
    disease = row['Disease'] #
    age = row['Age'] #
    gender = row['Gender'] #
    bp = row['Blood Pressure'] #
    chol = row['Cholesterol Level'] #
    outcome = row['Outcome Variable'] #
    
    # 'Yes'인 증상 추출
    symptoms = [s for s in symptom_cols if row[s] == 'Yes'] #
    if not symptoms:
        continue
    
    # 자연어 문장 생성 (RAG의 'text' 필드를 더 풍부하게 만듦)
    symptom_str = ', '.join(symptoms)
    
    # 변경 시작: text 필드에 환자 프로필 정보 추가
    # 이렇게 하면 임베딩 시 환자 프로필 정보도 함께 고려되어 검색 정확도를 높일 수 있습니다.
    text = f"증상: {symptom_str}이(가) 있고, 나이: {age}세, 성별: {gender}, 혈압: {bp}, 콜레스테롤: {chol}인 환자의 경우 {disease}일 수 있습니다."
    # 변경 끝

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