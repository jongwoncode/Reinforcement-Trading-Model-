import os
import logging
import abc
import collections
import threading
import time
import json
import numpy as np
from tqdm import tqdm
from environment import Environment
import tensorflow as tf
from agent import Agent
from network import LSTMNetwork
from visualizer import Visualizer
import utils


logger = logging.getLogger(utils.LOGGER_NAME)


class ReinforcementLearner():
    __metaclass__ = abc.ABCMeta

    def __init__(self, rl_method='rl', stock_code=None, chart_data=None, training_data=None,
                min_trading_price=100000, max_trading_price=10000000, balance=100000000, 
                num_steps=1, lr=0.0005, discount_factor=0.9, num_epoches=1000,
                start_epsilon=1, value_network=None, policy_network=None, output_path='', reuse_models=True):
        # 인자 확인
        assert min_trading_price > 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price
        assert balance > 100000
        assert num_steps > 0
        assert lr > 0
        # 강화학습 설정
        self.rl_method = rl_method
        self.discount_factor = discount_factor
        self.num_epoches = num_epoches
        self.start_epsilon = start_epsilon
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment, balance, min_trading_price, max_trading_price)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        # 로그 등 출력 경로
        self.output_path = output_path

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        self.environment.reset()                            # 환경 초기화
        self.agent.reset()                                  # 에이전트 초기화
        self.visualizer.clear([0, len(self.chart_data)])    # 가시화 초기화
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.                                      # 훈련 loss
        self.itr_cnt = 0                                    # 반복 횟수
        self.exploration_cnt = 0                            # 탐험 횟수(epsilon = 1)
        self.batch_size = 0
    '''
    # training_data 예시 : ./Test.ipynb -> 전처리 결과 예시
    # agent satae        : 보유자금 대비 포지션 보유 비율, 손익, 평균 수익률, 현재 포지션 
    ex) sample = ['close_ema15_ratio', ..., 'std20_mean20_ratio', ratio_hold, profitloss, avg_return, position] 
    '''
    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None
    # ActorCritic Class와 A2C Class에서 각각 method 구현
    @abc.abstractmethod
    def get_batch(self):
        pass

    def fit(self):
        # 배치 학습 데이터 생성 및 손실 초기화
        x, y_value, y_policy = self.get_batch()
        self.loss = None
        if len(x) > 0:
            loss = 0
            # 가치, 정책 신경망 갱신
            loss += self.value_network.train_on_batch(x, y_value)
            loss += self.policy_network.train_on_batch(x, y_policy)
            self.loss = loss

    def visualize(self, epoch_str, num_epoches, epsilon):
        pass 


    def run(self, learning=True):
        info = (f'[{self.stock_code}] RL:{self.rl_method} NET:{self.net} '
                f'LR:{self.lr} DF:{self.discount_factor} ')
        logger.debug(info)
        # 시작 시간
        time_start = time.time()
        # 가시화 준비 및 저장 폴더 준비
        self.visualizer.prepare(self.environment.chart_data, info)
        self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary_{self.stock_code}')
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else :
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0
        # 에포크 반복
        for epoch in tqdm(range(self.num_epoches)):
            time_start_epoch = time.time()
            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()
            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = self.start_epsilon * (1 - (epoch / (self.num_epoches - 1)))
            else:
                epsilon = self.start_epsilon
            # Epoch마다 반복.
            for i in tqdm(range(len(self.training_data)), leave=False) :
                next_sample = self.build_sample()   # 샘플 생성
                if next_sample is None:
                    break
                # Batch 생성 : num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps :
                    continue
                # 가치, 정책 신경망 예측
                pred_value = self.value_network.predict(list(q_sample))
                pred_policy = self.policy_network.predict(list(q_sample))
                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = self.agent.decide_action_by_policy(pred_policy, epsilon)
                # 결정한 행동을 수행하고 보상 획득
                reward = self.agent.act(action, confidence)
                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(reward)
                self.memory_value.append(pred_value)
                self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)
                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0
            # 에포크 종료 후 학습
            if learning :
                self.fit()

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(self.num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            logger.debug(f'[{self.stock_code}][Epoch {epoch_str}/{self.num_epoches}] '
                        f'Epsilon:{epsilon:.4f} #Expl.:{self.exploration_cnt}/{self.itr_cnt} '
                        f'#Buy:{self.agent.num_buy} #Sell:{self.agent.num_sell} #Hold:{self.agent.num_hold} '
                        f'#Stocks:{self.agent.num_stocks} PV:{self.agent.portfolio_value:,.0f} '
                        f'Loss:{self.loss:.6f} ET:{elapsed_time_epoch:.4f}')

            # 에포크 관련 정보 가시화
            if self.num_epoches == 1 or (epoch + 1) % int(self.num_epoches / 10) == 0:
                self.visualize(epoch_str, self.num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance :
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logger.debug(f'[{self.stock_code}] Elapsed Time:{elapsed_time:.4f} '
                f'Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}')

    def save_models(self) :
        if self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

    def predict(self) :
        # 에이전트 초기화
        self.agent.reset()
        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)
        
        result = []
        while True :
            # 샘플 생성
            next_sample = self.build_sample()
            if next_sample is None:
                break
            # num_steps만큼 샘플 저장
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue
            # 정책 신경망 예측
            pred_policy = self.policy_network.predict(list(q_sample))
            # 신경망에 의한 행동 결정
            action, confidence, _ = self.agent.decide_action_by_policy(pred_policy, 0)
            result.append((self.environment.observation[0], int(action), float(confidence)))

        with open(os.path.join(self.output_path, f'pred_{self.stock_code}.json'), 'w') as f :
            print(json.dumps(result), file=f)
        return result


class ActorCriticLearner(ReinforcementLearner) :
    def __init__(self, *args, value_network_path=None, policy_network_path=None, **kwargs) :
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        # Actor, Critic Network의 상단 공유
        self.shared_network = LSTMNetwork.get_network_head(input=self.num_features)
        # Critic Network(Value Network) 초기화
        self.value_network = LSTMNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, 
                                        shared_network=self.shared_network, lr=self.lr, num_steps=self.num_steps, 
                                        activation='linear', loss='mse')                                
        # Actor Network(Policy Network) 초기화
        self.policy_network = LSTMNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, 
                                         shared_network=self.shared_network, lr=self.lr, num_steps=self.num_steps, 
                                         activation='softmax', loss='categorical_crossentropy')

        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)                                    
        
        if self.reuse_models and os.path.exists(self.value_network_path) :
            self.value_network.load_model(model_path=self.value_network_path)
        
    def get_batch(self):
        memory = zip(reversed(self.memory_sample),
                     reversed(self.memory_action),
                     reversed(self.memory_value),
                     reversed(self.memory_policy),
                     reversed(self.memory_reward))
            
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory) :
            x[i] = sample
            # (다음 수익률 - 현재 수익률) + (최종 수익률 - 현재 수익률)
            r = (reward_next - reward) + (self.memory_reward[-1] - reward)
            y_value[i, :] = value
            y_value[i, action] = r + self.discount_factor * value_max_next
            y_policy[i, :] = policy
            y_policy[i, action] = utils.softmax(r)
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy



class A2CLearner(ActorCriticLearner) :
    def __init__(self, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = (reward_next - reward) + (self.memory_reward[-1] - reward) 
            y_value[i, :] = value
            y_value[i, action] = r + self.discount_factor * value_max_next
            advantage = y_value[i, action] - y_value[i].mean()
            y_policy[i, :] = policy
            y_policy[i, action] = utils.softmax(advantage)
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy

