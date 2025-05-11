# 인공지능 프로젝트

### Anaconda 사용법

* anaconda는 쉽게 말해서 Python용 docker이다
* 가상환경에 python 서버를 돌리는 것과 마찬가지

1. Anoconda를 다운로드 한다
2. Anoconda Prompt를 실행
3. conda env list를 치면 처음엔 base라고 뜰 것이다
4. 새로운 환경을 만들기 위해 conda create -n "ENV이름" //나의 경우 pyhome이라고 함 
5. (conda) activate ENV이름으로 하면 왼쪽에 (ENV이름)으로 바뀐게 보일 거임
6. conda install pip 이후 나머지는 pip install로 패키지 설치
7. pip install ipykernel
8. python -m ipykernel install --user --name ENV이름 --display-name "ENV이름"
9. conda install jupyter notebook -y
10. conda install nb_conda -y
11. pytorch의 경우 명령어가 os마다 다르므로 공식문서에서 명령어 확인하고 복붙

### 해야할일
* ~~자동으로 벡터 DB에서 조회해서 알아서 찾기~~
* ~~그걸 찾은 걸 기반으로 답변하게 하는 것~~
* 프롬프트 엔지니어링으로 정형화된 답변을 제공하게 만들 것 
* 인공지능 프로젝트의 목적성을 명확히 하기 (임상 데이터를 바탕으로 간이 진단 LLM을 만들 것인가?)
* 성능 측정 기준에 대하여 설정하기
