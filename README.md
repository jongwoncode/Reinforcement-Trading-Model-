# 강화학습을 활용한 트레이딩 봇

# 1. 프로젝트 정보
    
* 😀 프로젝트 구성원 : 이종원(개인프로젝트)
* 📆 프로젝트 기간  : 2022.12.15 ~ 2023.01.08
* 💻 사용 모델     : 강화학습 알고리즘(A3C), 신경망 모델(LSTM-DNN custom model)
* 🤖 주요 사용 기술  : python, tensorflow, keras, multi-processing

<br>

# 2. 목차
* **1. 프로젝트 정보**
* **2. 목차**
* **3. A3C 알고리즘 설명**
* **4. LSTMDNN 모델 설명 및 기본 모델(DNN)과 비교**
* **5. 파일 구조 및 경로 설명**
* **6. 훈련 및 테스트 결과 확인**
* **7. cmd 입력**
* **8. 고찰**


<br>

# 3. A3C(Asynchronous Advantage Actor-Critic) 

## 3.1 **A3C(Asynchronous Advantage Actor-Critic) 알고리즘**

* **Multi A2C(Advantage Actor Critic) Agent를 Asynchronous한 방식으로 훈련하는 알고리즘.**
    * **AC(Actor Critic)**

        * **Actor Network**와 **Critic Network**로 이루어진 강화학습 알고리즘

            - **Actor Network** : 최적의 action을 선택하는 것에 목적성을 둔 Network
            - **Critic Network** : $Q(s, a)$ 함수를 정확히 평가하는 것에 목적성을 둔 Network.
        
        <br>
        
        * **"Policy Iteration"** 방식의 강화학습 알고리즘

            - **Actor Network** 로 최적의 action을 선택할 수 있게 모델을 업데이트 하고 **(정책 발전)**,
            - **Critic Network** 를 통해 $Q(s, a)$를 정확히 추정하도록 모델을 업데이트 하여 policy의 신뢰도를 확보하는 알고리즘. **(정책 평가)**

        <br>

        * **AC의 정책신경망(Actor Network) 파라미터 업데이트 수식**
            $$
                \theta_{t+1}:= \theta_{t} + \alpha\left [\bigtriangledown_{\theta}\log\pi_{\theta}(a|s)Q_{w}(s,a))\right ]\\

                \theta : \;actor\; network\; parameter \\
                \qquad w : \;value\; network\; parameter
            $$ 
            - $Loss function$ = $Policy\;Network$ 출력의 $Cross\;Entropy$ $\times$ $Critic\;Network$의 출력
    
    <br>

    * **A2C(Advantage Actor Critic)**

        * **AC**의 가중치 업데이트 공식이 $Q(s,a)$함수 $(=Critic Network)$ 에 따라 변동이 심해 $Baseline$의 개념을 도입하여 학습의 안전성을 확보.

        <br>

        * **Baseline으로 Value function의 도입**
            - 가치 함수 $V(s)$는 상태$(s)$ 마다 다르지만, 행동마다 다르지 않기 때문에 효율적으로 큐 함수 $Q(s, a)$의 분산을 효율적으로 줄일 수 있음.
        
        <br>

        * **Advantage의 정의**
            $$
                Advantage = Action\;Value\;function(Q\;function) - Value\;function \\ 
                A(s_{t}, a_{t}) = Q_{w}(s_{t}, a_{t})-V_{v}(s_{t})
            $$

        * **TD error**
            - $advantage$ 에서 $Q$함수와 $V$함수를 각각 근사하려면 비효율적임. $TD\; error$의 개념을 적용하여 $Q$함수를 $V$와 $reward$로 변환
            $$
                TD\; Error : \delta_{v} = R_{t+1} + rV_{v}(S_{t+1}) -V_v(S_t)
            $$
            $$
                Critic\;parameter\; update : MSE =(R_{t+1} + rV_{v}(S_{t+1}) -V_v(S_t))^{2} = \delta_{v}^{2}
            $$
            $$
                Actor\; parameter\; update :  \theta_{t+1}:= \theta_{t} + \alpha\left [\bigtriangledown_{\theta}\log\pi_{\theta}(a|s)\delta_{v})\right ]
            
            $$

    <br>

* **On-policy 방식 알고리즘**
    * 주어진 $state$에 대해 $action$을 결정하는 $policy$가 존재하는 알고리즘.
    * $DQN$의 $off\; policy$ 부분을 개선.

<br>

* **Global Network & Replay Memory**
    * 각 Actor Learner는 각기 다른 환경에서 학습을 진행하기 때문에 학습 샘플들의 연관성을 낮을 수 있는 장점이 있음.
        - $DQN$의 $Replay\;Memory$의 아이디어를 활용하였고, 현재 $policy$ 기반의 의사결정에 의한 데이터를 저장할 수 있음.
    * 각 $Actor Learner$는 일정 횟수$(batch, time step)$를 탐색 하고 탐색 데이터를 활용하여 $global\; network$를 업데이트 한뒤 자신$(local\; network)$을 $global\; network$로 업데이트 함.  


<br>

# 4. LSTMDNN 모델 설명 및 기본 모델(DNN)과 비교


<br>

<br>


# 5. 파일 구조 및 경로 설명



<br>

<br>


# 6. 훈련 및 테스트 결과 확인





<br>


<br>

# 7. cmd (argparser)

|입력인자|설명|type|default|
|--|--|--|--|
|$--name$|로그 파일의 이름|$string$|파일실행시간|
|$--code$|투자 종목 코드|$string$|'005380' (=현대자동차)|
|$--model$|신경망 모델 설정|$choice$ = ['LSTMDNN', 'DNN']|'LSTMDNN'|
|$--mode$|학습 모드|$choice$=['train', 'test', 'update', 'monkey']|'train'|
|$--start_date$|훈련 데이터 시작일|$string$|'20180601'|
|$--end_date$|훈련 데이터 마지막일|$string$|'20221220'|
|$--lr$|learning rate|$float$|0.0001|
|$--n_steps$|LSTMDNN Network의 n_steps|$int$|10|
|$--balance$|초기 잔고|$int$|100000000|

<br>

> ## main.py 내 입력 인자

```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=utils.get_time_str())
    parser.add_argument('--code', type=str, default='005380')
    parser.add_argument('--model', choices=['LSTMDNN', 'DNN'], default='LSTMDNN')
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'monkey'], default='train')
    parser.add_argument('--start_date', default='20180601')
    parser.add_argument('--end_date', default='20221220')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--balance', type=int, default=100000000)
    args = parser.parse_args()

```

<br>

> ## 예시
* 종목코드 '005380'을 n_step이 10인 LSTMDNN 신경망으로 훈련 시킴. learning rate는 0.0001. 훈련 데이터는 20200101 ~ 20221220
```
python main.py --code 005380 --model LSTMDNN --mode train --start_date 20200101 --lr 0.0001 -- n_steps 10
```

* 종목코드 '005380'을 n_step이 10인 기존에 훈련된 LSTMDNN 신경망으로 업데이트 시킴. learning rate는 0.00006, 업데이트 데이터는 20180601~20221220
```
python main.py --code 005380 --model LSTMDNN --mode update --start_date 20200101 --lr 0.00006
```

* 종목코드 '005380'을 DNN 신경망으로 훈련 시킴. learning rate는 0.0001, 업데이트 데이터는 20180601~20221220
```
python main.py --code 005380 --model DNN
```


<br>


<br>

# 8. 고찰 