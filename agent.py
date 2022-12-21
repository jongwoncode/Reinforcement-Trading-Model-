import environment
import numpy as np
import utils

'''
Agent : 선물 거래를 염두한 행동 전략(long-short 배팅 가능)
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
        1. 훈련 단계에서는 Slippage 적용.           (아직 미반영)
        2. 실제 거래 단계에서는 api를 통해 실제 balance를 조회하여 적용.
'''
class Agent :
    # agent state : [포지션/자금 비율, 손익, 평균 수익률, 현재 포지션]
    STATE_DIM = 4
    # --- 포지션  
    SHORT_POSITION = 0          # 숏 포지션
    NONE_POSITION  = 1          # 포지션 없음
    LONG_POSITION  = 2          # 롱 포지션

    # --- 수수료 및 세금  
    TRADING_CHARGE = 0.00015    # 매매 수수료(0.015%)
    TRADING_TAX    = 0.0025     # 세금 (0.25%)

    # --- 행동 종류 
    ACTION_SHORT = 0            # 숏 (= 공매도) 
    ACTION_HOLD  = 1            # 숏, 롱 안함
    ACTION_LONG  = 2            # 롱
    ACTIONS = [ACTION_SHORT, ACTION_HOLD, ACTION_LONG]    # action space
    NUM_ACTIONS = len(ACTIONS)                            # action space length

    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price) :
        # --- 환경 설정 --- # 
        self.environment = environment          # 환경 객체 로드
        self.initial_balance = initial_balance  # 초기 자본금

        # 최소 단일 매매 금액, 최대 단일 매매 금액
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price
        
        # Agent 클래스의 속성
        self.balance = initial_balance  # 현재 현금 잔고
        self.num_stocks = 0             # 보유 주식 수
        self.portfolio_value = 0        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.num_long = 0                # 매수 횟수
        self.num_short = 0               # 매도 횟수
        self.num_hold = 0               # 관망 횟수

        # Agent 클래스의 상태
        self.ratio_hold = 0             # 주식 보유 비율
        self.profitloss = 0             # 현재 손익
        self.avg_position_price = 0          # 주당 매수 단가
        self.position = 1               # 현재 포지션
    
    def reset(self) :
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

    def set_balance(self, balance) :
        self.initial_balance = balance

    # agent state space를 반환해주는 함수
    # agent state : [포지션 보율 비율, 손익, 평균 수익률, 현재 포지션]
    def get_states(self) :
        self.ratio_hold = np.abs(self.num_stocks*self.environment.get_price())/ self.portfolio_value
        avg_return = 0
        if self.position == Agent.LONG_POSITION :
            avg_return = (self.environment.get_price() / self.avg_position_price) - 1
        elif self.position == Agent.SHORT_POSITION :
            avg_return = (self.avg_position_price / self.environment.get_price()) - 1

        return (self.ratio_hold, self.profitloss, avg_return, self.position)

    # policy network를 통한 action 결정(Actor Network)
    def decide_action_by_policy(self, pred_policy, epsilon) :
        confidence = .5
        if pred_policy is None:
            epsilon = 1                         # 예측 값이 없을 경우 탐험
        elif (pred_policy == np.max(pred_policy)).all() :    # 값이 모두 같은 경우 탐험
            epsilon = 1
        # 탐험 결정
        if np.random.rand() < epsilon :
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else :
            exploration = False
            action = np.argmax(pred_policy)
        
        if pred_policy is not None :
            confidence = pred_policy[action]
        return action, confidence, exploration

    # value network를 통한 action 결정 (Critic Network)
    def decide_action_by_value(self, pred_value, epsilon) :
        confidence = .5
        if pred_value is None :
            epsilon = 1
        elif (pred_value == np.max(pred_value)).all() :    # 값이 모두 같은 경우 탐험
            epsilon = 1
        # 탐험 결정
        if np.random.rand() < epsilon :
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred_value)

        if pred_value is not None:
            confidence = utils.sigmoid(pred_value[action])
        return action, confidence, exploration

    # 결정된 Action(Long, Short)을 수행할 수 있는 최소 조건을 확인
    def validate_action(self, action) :
        if action == Agent.ACTION_LONG :
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) :
                return False
        elif action == Agent.ACTION_SHORT :
            # 적어도 1주를 short할 수 있는지 확인
            if self.portfolio_value < self.environment.get_price() * (1 + self.TRADING_CHARGE) :
                return False
        return True

    # Action을 수행할 수 있을 때 진입 포지션의 양을 반환해주는 함수.
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence) :
            return self.min_trading_price
        added_trading_price = max(
            min(int(confidence*(self.max_trading_price-self.min_trading_price)), self.max_trading_price-self.min_trading_price), 0)
        trading_price = self.min_trading_price + added_trading_price
        return max(int(trading_price / self.environment.get_price()), 1)
    
    #  action : './logic/action_logic.jpg' 참조
    def act(self, action, confidence):
        if not self.validate_action(action) :
            action = Agent.ACTION_HOLD
        # 현재 가격 수집
        curr_price = self.environment.get_price() 

        # (1) 현재 NONE 포지션(num_stocks == 0)
        if self.position == Agent.NONE_POSITION :
            # (1.1) HOLD 
            if action == Agent.ACTION_HOLD :
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
                    if action == Agent.ACTION_LONG :     
                        self.num_stocks += trading_unit
                    # (1.2.5.2) SHORT 진입 : 보유 주식수 차감 
                    elif action == Agent.ACTION_SHORT :
                        self.num_stocks -= trading_unit

        # (2) 현재 LONG 포지션(num_stocks > 0)
        elif self.position == Agent.LONG_POSITION :
            # (2.1) LONG 진입
            if action == Agent.ACTION_LONG :
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
            elif action == Agent.ACTION_SHORT :
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
            elif action == Agent.ACTION_HOLD :
                self.num_hold += 1              # (2.3.1) 관망 횟수 증가



        # (3) 현재 SHORT 포지션(num_stocks < 0)
        elif self.position == Agent.SHORT_POSITION :
            # (3.1) LONG 진입
            if action == Agent.ACTION_LONG :
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
            elif action == Agent.ACTION_SHORT :
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
            elif action == Agent.ACTION_HOLD :
                self.num_hold += 1              # (3.3.1) 관망 횟수 증가
        # (4) 포지션 업데이트
        if self.num_stocks > 0 :
            self.position = Agent.LONG_POSITION
        elif self.num_stocks < 0 :
            self.position = Agent.SHORT_POSITION
        else :
            self.position = Agent.NONE_POSITION

        # (5) 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * abs(self.num_stocks)
        self.profitloss = self.portfolio_value / self.initial_balance - 1
        return self.profitloss