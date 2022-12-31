import numpy as np

'''
Environment : 선물 거래를 염두한 행동 전략(long-short 배팅 가능)
    [Long-Short position]
        1. [훈련] Action에 의해 결정된 short position 물량이 현재 long 포지션의 물량을 넘으면 long 포지션을 모두 청산(close)하고 남은 물량을 open short
        2. [실전] Model을 수정.(+ 고립, 격리 반영)

    [state]
        1. 포지션_자금 비율      : postion_(t)/portfolio_value_(t)
        2. 손익                 : portfolio_value_(t)/portfolio_value_(0)
        3. 현재 포지션           : SHORT = 0, LONG = 1
        4. 평균 수익률
            * [LONG position]  close_price_(t) - mean_position_price_(t)
            * [SHORT position] mean_position_price_(t) - close_price_()
        
    [Slippage]
        1. 훈련 단계에서는 Slippage 적용. (현재 미적용)     
        2. 실제 거래 단계에서는 api를 통해 실제 balance를 조회하여 적용.
'''

class Environment() :
    # agent balance state : [포지션/자금 비율, 손익, 평균 수익률, 현재 포지션]
    B_STATE_DIM = 4
    # 포지션 정보(매수, 무포지션, 공매도)
    SHORT_POSITION = 0 
    NONE_POSITION = 1
    LONG_POSITION = 2               
    # 수수료(0.015%), 세금(0.25%)
    TRADING_CHARGE = 0.00015 
    TRADING_TAX = 0.0025    
    # 행동 종류 (숏[공매도], 행동 안함, 롱[매수])
    ACTION_SHORT = 0
    ACTION_HOLD = 1
    ACTION_LONG = 2 
    # action space & length
    ACTIONS = [ACTION_SHORT, ACTION_HOLD, ACTION_LONG]
    NUM_ACTIONS = len(ACTIONS)                            
    CLOSE_IDX = 4
    def __init__(self, chart_data, training_data, n_steps, initial_balance, min_trading_price, max_trading_price) :
        #___초기 자본금 설정
        self.initial_balance = initial_balance

        #___최소/최대 단일 매매 금액 설정
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        #___chart : chart 정보
        self.chart_data = chart_data
        self.training_data = training_data
        self.n_steps = n_steps
        self.observation = None
        self.idx = -1

        #___balance : 잔고 내역 및 거래 정보
        self.balance = initial_balance   # 현재 현금 잔고
        self.num_stocks = 0              # 보유 주식 수
        self.portfolio_value = 0         # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.num_long = 0                # 매수 횟수
        self.num_short = 0               # 매도 횟수
        self.num_hold = 0                # 관망 횟수

        #___balance : agent의 state 정보
        self.ratio_hold = 0              # 주식 보유 비율
        self.profitloss = 0              # 현재 손익
        self.avg_position_price = 0      # 주당 매수 단가
        self.position = 1                # 현재 포지션

    def reset(self):
        self.observation = None
        self.idx = -1
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.num_stocks = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_position_price = 0
        self.position = 1
        self.num_long = 0
        self.num_short = 0
        self.num_hold = 0

    def observe(self):
        if len(self.chart_data) > self.idx + 1 :
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self) :
        return self.observation[self.CLOSE_IDX]

    # 결정된 Action(Long, Short)을 수행할 수 있는 최소 조건을 확인
    def validate_action(self, action) :
        if action == Environment.ACTION_LONG :
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.get_price() * (1 + self.TRADING_CHARGE) :
                return False
        elif action == Environment.ACTION_SHORT :
            # 적어도 1주를 short할 수 있는지 확인
            if self.portfolio_value < self.get_price() * (1 + self.TRADING_CHARGE) :
                return False
        return True

    # Action을 수행할 수 있을 때 진입 포지션의 양을 반환해주는 함수.
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence) :
            return self.min_trading_price
        added_trading_price = max(min(int(confidence*(self.max_trading_price-self.min_trading_price)), 
                                            self.max_trading_price-self.min_trading_price), 0)
        trading_price = self.min_trading_price + added_trading_price
        return max(int(trading_price / self.get_price()), 1)
    
    # action을 수행하고 환경 정보를 업데이트
    # input : policy에 출력된 action과 confidence
    # output : 현재까지 수익률(reward & return) 
    def act(self, action, confidence):
        # action을 수행할 수 있는지 잔고 확인 -> 수행할 수 없다면 HOLD 포지션.
        if not self.validate_action(action) :
            action = Environment.ACTION_HOLD
        curr_price = self.get_price() 

        # (1) 현재 NONE 포지션(num_stocks == 0)
        if self.position == Environment.NONE_POSITION :
            # (1.1) HOLD 
            if action == Environment.ACTION_HOLD :
                # (1.1.1) 관망 횟수 증가
                self.num_hold += 1
            # (1.2) LONG or SHORT 진입
            else : 
                # (1.2.1) 진입 유닛 설정 
                trading_unit = self.decide_trading_unit(confidence)
                # (1.2.2) 보유 현금 검증
                remain_balance =  self.balance - (curr_price * (1 + self.TRADING_CHARGE) * trading_unit)
                # (1.2.3) 보유 현금 부족시 진입 금액&유닛 재산정
                if remain_balance < 0 :
                    possible_amount = min(self.balance, self.max_trading_price)
                    trading_unit = int(possible_amount / (curr_price * (1 + self.TRADING_CHARGE)))
                # (1.2.4) 진입 금액 산정
                trading_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
                # (1.2.5) 진입 금액 존재시 정보 갱신
                if trading_amount > 0 :
                    self.avg_position_price = curr_price    # 평균 포지션 가격 업데이트
                    self.balance -= trading_amount          # 보유 현금을 갱신
                    self.num_long += 1                      # long 횟수 증가
                    # (1.2.5.1) LONG 진입 : 보유 주식수 추가
                    if action == Environment.ACTION_LONG :     
                        self.num_stocks += trading_unit
                    # (1.2.5.2) SHORT 진입 : 보유 주식수 차감 
                    elif action == Environment.ACTION_SHORT :
                        self.num_stocks -= trading_unit

        # (2) 현재 LONG 포지션(num_stocks > 0)
        elif self.position == Environment.LONG_POSITION :
            # (2.1) LONG 진입
            if action == Environment.ACTION_LONG :
                trading_unit = self.decide_trading_unit(confidence)                                 # (2.1.1) 진입 유닛 설정 
                remain_balance = self.balance - (curr_price * (1 + self.TRADING_CHARGE) * trading_unit)   # (2.1.2) 보유 현금 검증
                # (2.1.3) 보유 현금 부족시 진입 금액&유닛 재산정
                if remain_balance < 0 : 
                    possible_amount = min(self.balance, self.max_trading_price)
                    trading_unit = int(possible_amount / (curr_price * (1 + self.TRADING_CHARGE)))
                trading_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit              # (2.1.4) 진입 금액 산정
                # (2.1.5) 진입 금액 존재시 정보 갱신
                if trading_amount > 0 :
                    self.avg_position_price = (self.avg_position_price * self.num_stocks + curr_price * trading_unit) \
                                                / (self.num_stocks + trading_unit)      # 평균 포지션 가격 업데이트
                    self.balance -= trading_amount                                      # 보유 현금을 갱신
                    self.num_long += 1                                                  # long 횟수 증가
                    self.num_stocks += trading_unit                                     # 보유 주식수 추가
                
            # (2.2) SHORT 진입
            elif action == Environment.ACTION_SHORT :
                # (2.2.1) 진입 유닛 설정 
                trading_unit = self.decide_trading_unit(confidence)
                # (2.2.2) 보유 물량 검증
                remain_unit = self.num_stocks - trading_unit
                # (2.2.3) 보유 물량 부족 -> 보유 현금 확인
                if remain_unit < 0 :
                    # (2.2.3.2) 잔여 balance 계산(= 기존 balance + 기존 매수 물량 매도 금액 - 신규 공매도 금액)
                    add_amount = (curr_price * (1 - self.TRADING_CHARGE) * self.num_stocks) 
                    remain_balance = self.balance + add_amount - (curr_price * (1 + self.TRADING_CHARGE) * abs(remain_unit))
                    # (2.2.3.1) 보유 현금 부족시 진입 금액&유닛 재산정
                    if remain_balance < 0 :
                        possible_amount = min(self.balance + add_amount, self.max_trading_price)
                        trading_unit = int(possible_amount / (curr_price * (1 + self.TRADING_CHARGE)))
                    # (2.2.3.2) 기존 물량(self.num_stocks) 모두 매도 + 보유 현금으로 나머지 Short 포지션 진입
                    trading_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
                    self.balance = self.balance + add_amount - trading_amount  # balance 정보 갱신(*)
                # (2.2.4) 보유 물량 부족 X
                else :
                    trading_amount = curr_price * (1 - self.TRADING_CHARGE) * trading_unit
                    self.balance += trading_amount  # balance 정보 갱신(*)

                # (2.2.5) 진입 금액 존재시 정보 갱신
                if trading_amount > 0 :
                    self.num_stocks -= trading_unit         # 보유 주식수 차감
                    # 평균 포지션 가격 & 보유 현금 갱신
                    if self.num_stocks > 0 :
                        self.avg_position_price = self.avg_position_price
                    elif self.num_stocks < 0 :
                        self.avg_position_price = curr_price
                    else :
                        self.avg_position_price = 0 
                    self.num_short += 1                     # short 횟수 증가
            # (2.3) HOLD 
            elif action == Environment.ACTION_HOLD :
                self.num_hold += 1              # (2.3.1) 관망 횟수 증가

        # (3) 현재 SHORT 포지션(num_stocks < 0)
        elif self.position == Environment.SHORT_POSITION :
            # (3.1) LONG 진입
            if action == Environment.ACTION_LONG :
                # (3.2.1) 진입 유닛 설정 
                trading_unit = self.decide_trading_unit(confidence)
                # (3.2.2) 공매도 물량 초과 확인
                remain_unit = self.num_stocks + trading_unit
                # (3.2.3) 공매도 물량 소진 O + 초과 매수
                if remain_unit > 0 :
                    # (3.2.3.1) 잔여 balance 계산(=기존 balance + 공매도 포지션 정리 금액 - remain_unit * 현재 가격)
                    add_amount = curr_price * (1 - self.TRADING_CHARGE) * abs(self.num_stocks)
                    remain_balance = self.balance + add_amount - (curr_price * (1 + self.TRADING_CHARGE) * remain_unit)
                    # (3.2.3.2) 보유 현금 부족시 진입 금액&유닛 재산정
                    if remain_balance < 0 :
                        possible_amount = min(self.balance + add_amount, self.max_trading_price)
                        trading_unit = int(possible_amount / (curr_price * (1 + self.TRADING_CHARGE)))
                    # (3.2.3.3) 기존 공매도 물량(self.num_stocks) 모두 매수 + 보유 현금으로 나머지 Long 포지션 진입
                    trading_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
                    self.balance = self.balance + add_amount - trading_amount  # balance 정보 갱신(*)
                # (3.2.4) 공매도 물량 소진 X
                else :
                    trading_amount = curr_price * (1 - self.TRADING_CHARGE) * trading_unit
                    self.balance += trading_amount  # balance 정보 갱신(*)

                # (3.2.5) 진입 금액 존재시 정보 갱신
                if trading_amount > 0 :
                    self.num_stocks += trading_unit         # 보유 주식수 추가
                    # 평균 포지션 가격 & 보유 현금 갱신
                    if self.num_stocks < 0 :
                        self.avg_position_price = self.avg_position_price
                    elif self.num_stocks > 0 :
                        self.avg_position_price = curr_price
                    else :
                        self.avg_position_price = 0
                    self.num_long += 1                      # long 횟수 증가

            # (3.2) SHORT 진입
            elif action == Environment.ACTION_SHORT :
                trading_unit = self.decide_trading_unit(confidence)                                       # (3.1.1) 진입 유닛 설정 
                remain_balance = self.balance - (curr_price * (1 + self.TRADING_CHARGE) * trading_unit)   # (3.1.2) 보유 현금 검증
                # (3.1.3) 보유 현금 부족시 진입 금액&유닛 재산정
                if remain_balance < 0 :
                    possible_amount = min(self.balance, self.max_trading_price)
                    trading_unit = int(possible_amount / (curr_price * (1 + self.TRADING_CHARGE)))
                trading_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit                    # (3.1.4) 진입 금액 산정
                # (3.1.5) 진입 금액 존재시 정보 갱신
                if trading_amount > 0 :
                    self.avg_position_price = (self.avg_position_price * abs(self.num_stocks) + curr_price * trading_unit) \
                                                / (abs(self.num_stocks) + trading_unit)      # 평균 포지션 가격 업데이트
                    self.balance -= trading_amount                                      # 보유 현금을 갱신
                    self.num_short += 1                                                 # short 횟수 증가
                    self.num_stocks -= trading_unit                                     # 보유 주식수 차감
            # (3.3) HOLD 
            elif action == Environment.ACTION_HOLD :
                self.num_hold += 1              # (3.3.1) 관망 횟수 증가
        # (4) 포지션 업데이트
        if self.num_stocks > 0 :
            self.position = Environment.LONG_POSITION
        elif self.num_stocks < 0 :
            self.position = Environment.SHORT_POSITION
        else :
            self.position = Environment.NONE_POSITION

        # (5) 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * abs(self.num_stocks)
        self.profitloss = self.portfolio_value / self.initial_balance - 1
        return self.profitloss


    # input  : agent's action space
    # output : (observed_chart, observed_balance, reward, done, info)
    def step(self, action=None, policy=None) :
        observation = self.observe()
        # 다음 훈련 데이터가 없을 경우.
        if (np.array(observation) == None).all() :
            done = True
            return None, None, 0, done
        # 훈련 시작 전 초기 데이터 반환
        if action == None :
            avg_return = (self.avg_position_price / self.get_price()) - 1
            c_next_state = self.training_data[self.idx]
            b_next_state = (self.ratio_hold, self.profitloss, avg_return, self.position)
            done = False
            return c_next_state, b_next_state, 0, done
        
        # agent의 행동에 대해서 다음 환경 정보를 반환
        else :
            # action에 대한 신뢰도 계산
            confidence = policy[action]
            # 행동 수행 및 보상 출력
            reward = self.act(action, confidence)
            # 현재 종가 대비 평균 수익률
            avg_return = (self.avg_position_price / self.get_price()) - 1
            # chart state, balance state 계산
            c_next_state = self.training_data[self.idx]
            b_next_state = (self.ratio_hold, self.profitloss, avg_return, self.position)
            done = False
            # 전체 손실이 초기 투자금 태비, -50% 이면 epoch 종료. 
            if self.portfolio_value < self.initial_balance*0.5 :
                done = True
            return c_next_state, b_next_state, reward, done
  

