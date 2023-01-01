# 강화학습을 활용한 트레이딩 봇

## 1. 프로젝트 정보
    
* 😀 프로젝트 구성원 : 이종원(개인프로젝트)
* 📆 프로젝트 기간  : 2022.12.15 ~ 진행중 (2023.01.05 : 완료 예정 )
* 💻 사용 모델     : 강화학습 알고리즘(A3C), 신경망 모델(LSTM-DNN custom model)
* 🤖 주요 사용 기술  : python, tensorflow, keras, multi-processing

<br>

## 2. 프로젝트 개요
* **A3C(Asynchronous Advantage Actor-Critic) 모델 사용**
    - DQN의 장점을 차용하고 단점을 개선한 A3C모델 사용.
        + DQN 장점
            - 리플레이 메모리의 무작위 샘플 추출로 인한 샘플들의 상관관계 제거
        + DQN 단점
            - off-policy 방식
            - 리플레이의 메모리 데이터는 과거의 학습 정보
    - Multi Agent 방식 사용
        + Mutil Agent를 병렬 학습 하여 각 Agent마다 다른 상황에서 학습.
        + 주기적으로 local network를 global network로 업데이트. 
        + DQN의 replay memory의 랜덤 추출 기법의 아이디어를 적용.




