import textwrap
import chromadb
import numpy as np
import pandas as pd
import re
import json

from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from google.generativeai import types
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

client = genai.Client(api_key='AIzaSyCHcy8oO-XhjsLCTdzLB64t9XR01OanbpM')

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    EMBEDDING_MODEL_ID = "models/embedding-001"
    title = "Custom query"
    response = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=input,
        config=types.EmbedContentConfig(
          task_type="retrieval_document",
          title=title
        )
    )
    return [e.values for e in response.embeddings]

def preprocess_metadata(metadata):
    new_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, list):
            new_metadata[k] = ", ".join(map(str, v))
        else:
            new_metadata[k] = v
    return new_metadata

def batch_add(collection, documents, metadatas, ids, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )

def create_chroma_db(json_data, name):
    chroma_client = chromadb.Client()
    try:
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        print(f"Collection '{name}' already exists. Using existing collection.")
    except:
        db = chroma_client.create_collection(
            name=name,
            embedding_function=GeminiEmbeddingFunction()
        )
        print(f"Collection '{name}' created.")
        documents = [item["text"] for item in json_data]
        metadatas = [preprocess_metadata(item["metadata"]) for item in json_data]
        ids = [str(i) for i in range(len(json_data))]
        batch_add(db, documents, metadatas, ids, batch_size=100)
        print("Data added to the collection.")
   
    return db

try:
    with open('disease_rag_with_metadata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    db = create_chroma_db(data, "my_collection")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    db = None 

def get_relevant_passage(query, db, n_results=5):
  if db is None:
      return []
  results = db.query(query_texts=[query], n_results=n_results, include=['documents', 'metadatas'])
  passages = []
  for i in range(len(results['documents'][0])):
      doc = results['documents'][0][i]
      meta = results['metadatas'][0][i]
      passage_text = f"{doc} (나이: {meta.get('age', '정보 없음')}, 성별: {meta.get('gender', '정보 없음')}, 혈압: {meta.get('blood_pressure', '정보 없음')}, 콜레스테롤: {meta.get('cholesterol', '정보 없음')})"
      passages.append(passage_text)
  return passages

def make_prompt(query, relevant_passages):
  escaped = " ".join([p.replace("'", "").replace('"', "").replace("\n", " ") for p in relevant_passages])
  prompt = f"""
  당신은 사용자의 증상과 개인 프로필 정보를 기반으로 질병을 설명하고, **일상생활에서 할 수 있는 구체적이고 실용적인 조언을 상세하게 제공하는** 의료 상담 도우미입니다. 당신의 답변은 정보가 풍부하고 친절하며, **극단적이거나 심각한 질병을 직접적으로 진단하거나 추천하는 뉘앙스를 피해야 합니다.** 답변은 최소 100단어 이상으로 작성해 주세요.

  **단계별 지시사항:**
  1. 사용자 질문을 이해하고 핵심 증상(예: 기침, 콧물)을 정확하고 상세하게 파악하세요.
  2. 제공된 '관련 정보 (PASSAGE)'를 면밀히 검토하여 사용자 증상과 가장 밀접하게 일치하는 질병(들)을 식별하되, **데이터에 기반한 질병 연관성을 언급하되 불필요하게 심각성을 강조하지 마세요.**
  3. **여러 질병이 검색될 경우, 가장 일반적이거나 흔한 질환(예: 감기, 알레르기)을 우선적으로 상세히 설명하고, 그 다음으로 관련된 다른 질병들도 간략하게 제시하세요.**
  4. 나이, 성별, 혈압, 콜레스테롤 수치와 같은 환자 프로필 정보가 있다면, 이를 **답변의 서론 부분에 해당 질병이 특정 프로필의 환자에게서 관찰될 수 있는 '사례'로 자연스럽게 통합하여 설명의 깊이를 더하세요.** 질병 진단의 직접적인 근거로 오해되지 않도록 주의하세요.
  5. 답변은 정보가 풍부하고 명확하며 친절하게 작성하며, 다음 **상세 권장 출력 형식**을 따르되, **세부적인 구문은 모델의 자연스러운 생성에 맡기세요.**

  **상세 권장 출력 형식:**
  안녕하세요! [사용자 질문에서 파악된 증상]이(가) 있으시군요. 불편하시겠지만, 몇 가지 가능한 원인과 생활 속 대처법을 함께 알아보겠습니다.

  (선택적: 임상 데이터에 따르면, [관련 정보의 나이]세 [관련 정보의 성별] 환자 중 [관련 정보의 혈압] 혈압과 [관련 정보의 콜레스테롤] 콜레스테롤 수치를 가진 분들에게서 [해당 질병과 연결된 증상]이 관찰된 사례가 있습니다.) 이러한 증상들은 [관련 정보에서 찾은 가장 일반적이고 가능성 높은 질병]과 관련이 있을 수 있습니다.

  [질병에 대한 간략한 추가 설명 (2-3문장)]. 이 질병의 일반적인 경과나 특징에 대해 간략히 설명해 주세요.

  이럴 때는 다음과 같은 생활 습관 개선을 통해 증상 완화에 도움을 줄 수 있습니다:
  - **충분한 휴식:** 몸이 회복하는 데 필요한 시간을 주세요.
  - **수분 섭취:** 따뜻한 물, 차 등을 자주 마셔 목을 촉촉하게 유지하고 탈수를 예방하세요.
  - **실내 환경 관리:** 적절한 실내 습도를 유지하고 환기를 자주 해주세요.
  - **영양가 있는 음식 섭취:** 면역력 강화를 위해 비타민과 미네랄이 풍부한 음식을 드세요.
  - [추가적인 일반적인 조언 1 (예: 스트레스 관리, 가벼운 운동 등)]
  - [추가적인 일반적인 조언 2 (예: 마스크 착용, 손 씻기 등)]

  만약 [사용자 질문에서 파악된 증상] 외에 다른 불편한 증상이 있거나, 현재 증상이 나아지지 않고 오히려 심해진다면 [다른 관련 질병]일 수도 있습니다. (이때, 극단적인 질병은 가급적 언급하지 않거나, "드물게는 ~일 수도 있습니다"와 같이 조심스러운 표현을 사용하세요.)

  더 궁금한 점이 있으시면 언제든지 다시 질문해주세요. 항상 건강하시길 바랍니다.

  아래는 참고할 수 있는 임상 데이터입니다:
  - 사용자 질문 (QUESTION): \"{query}\"
  - 관련 정보 (PASSAGE): \"{escaped}\"

  **주의사항:**
  - 병원 방문 및 전문적인 상담을 직접적으로 권유하는 문구는 최종 답변에 포함하지 마세요.
  - PASSAGE에 영어 단어가 포함되어 있다면, 괄호 안에 한글 뜻을 함께 제공해 주세요.
  - **제공된 정보 내에서 '기침'과 '습진'의 연관성이 있더라도, '기침'이라는 증상에 더 일반적이고 흔한 질병(예: Common Cold, Influenza)이 있다면 이를 우선적으로 고려하여 답변하세요.**
  - **'말라리아'와 같이 심각한 질병은 사용자가 직접적으로 언급하지 않는 한, 일반적인 증상만으로는 추천하지 마세요.**

  ANSWER:
  """.format(query=query, relevant_passages=escaped)
  return prompt

app = FastAPI() #uvicorn server:app --reload 으로 실행

class UserQuery(BaseModel):
    query: str

@app.post("/dr.chat") #//http://localhost:8000/dr.chat"
async def consult_patient(user_query: UserQuery):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized.")

    query_text = user_query.query
    
    # Get relevant passages from ChromaDB
    relevant_passages = get_relevant_passage(query_text, db, 5) #
    
    # Generate prompt for Gemini model
    prompt = make_prompt(query_text, relevant_passages) #
    
    try:
        # Call Gemini model
        MODEL_ID = "gemini-2.0-flash"
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        
        # Extract the relevant part of the answer
        final_answer = response.text.split("ANSWER:")
        if len(final_answer) > 1:
            final_answer = final_answer[1].strip()
        else:
            final_answer = response.text.strip()
            
        return {"response": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from LLM: {e}")