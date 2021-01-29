### Dropout as a Bayesian Approximation- Representing Model Uncertainty in Deep Learning

---

#### Abstract

많은 관심을 받던 딥러닝 기술 (regression, classification) 은 모델 불확실성을 나타내지 못한다. 비교로, 베이지안 모델은 모델 불확실성을 추정하긴 하지만 많은 비용이 든다. 이 논문에서는 딥러닝에서의 드롭아웃이 가우시안 분포의 베이지안 추정과 근사한다는 것을 보여준다. 그러므로써 계산적으로 복잡함 없이 딥러닝의 불확실성을 나타낸다. 실험에서는 Regression 과 MNIST 를 사용한 classification 에서 다양한 네트워크를 평가한다. 그리고 현재의 방법과 비교하여 우리의 방법이 RMSE 와 predictive log-likelihood 의 상당한 향상을 보여주고, 끝으로 강화학습에서의 드롭아웃 불확실성을 사용에 대해서 설명한다.

#### 1. Introduction

Regression 과 classification 에서 쓰이는 standard 딥러닝 기술은 모델의 불확실성을 나타내지 않는다. Classification 에서는, 소프트맥스 아웃풋(파이프라인 마지막단계)의 예측 확률 값이 종종 모델의 confidence 라고 잘못 해석되어 진다. 모델이 소프트맥스의 높은 아웃풋을 가질지라도 불확실할 수 있기도 하다. 그림 1 을 보자.

![](./img/dropout_f1.png)

왼쪽 (소프트맥스 인풋), 오른쪽(소프트맥스 아웃풋), x* (트레이닝 데이터로 부터 멀리 떨어진데이터)

\- 트레이닝 데이터로부터 멀리 떨어진 데이터에 대해 높은 컨피던스와 함께 나타난다. x* 같은 경우 1의 확률로 분류되고 있다. 하지만 distribution 을 소프트맥스에 통과시킨 결과는 트레이닝 데이터로부터 멀리 떨어진 데이터에 대한 불확실성을 잘 반영해준다.

모델의 불확실성은 딥러닝에서 매우 중요하다. 예를 들어 우체국에서 문자를 분류하는 작업에서 모델이 높은 불확실성을 나타낸다면, 분류작업을 사람에게 넘겨줄 수 도 있다. 또한 강화학습에서도 중요한데, 에이전트는 불확실성을 가지고, explore 할 지, exploit 할 지를 결정할 수 있다. 최근 RL 은 Q-value function approximation 네트워크를 사용하고 있는데, 이 함수는 에이전트가 취한 각 행동의 퀄리티를 측정해준다. Epsilon greedy 는 확률을 가지고 best action 을 고르도록 할 때 종종 사용된다. 에이전트의 Q-value function 을 걸쳐 불확실성을 평가하면 Thompson sampling 과 같은 기술이 학습을 더 빨리하는데 사용될 수 있다.

우리는 뉴럴네트워크에서 드롭아웃이 잘 알려진 확률모델인 가우시안 프로세스의 베이지안 추정방법으로 해석 될 수 있다는 것을 보인다. (so, 계산적 복잡함과 test accuracy의 희생없이 모델의 불확실성에 대한 문제를 완화할 수 있다.)

이 논문에서, 드롭아웃 뉴럴네트워크로 부터 얻은 불확실성에 대해 평가하고, 다른 모델로 부터 얻은 불확실성과 비교한다. 

#### 2. Related Search

무한한 뉴럴네트워크 분포(weights에 걸친)는 가우시안 프로세스에 수렴한다는 것은 오래 전부터 알아왔다. 유한한 NN에 대한 분포는 베이지안 뉴럴네트워크로 연구되어왔는데, 이것은 오버피팅에 대한 robustness 를 제공하지만 추정과 계산적 비용의 과제가 존재한다. Variational Inference (VI) 가 이런 모델에 적용되어 왔다. 이상, VI 의 최근 연구에 대한 설명.

#### 3. Dropout as a Bayesian Approximation

우리는 모든 weight layer에 드롭아웃을 적용한 비선형, 임의의 깊이의 뉴럴네트워크가 수학적으로 확률적 가우시안 프로세스의 근사치와 동일하다는 것을 보인다. 드롭아웃이 (근사하는 분포)와 딥 가우시안 프로세스의 posterior 사이의 KL-Divergence를 줄인다는 것도 보인다.

\+ 드롭아웃, 가우시안 프로세스, VI 에 대한 설명은 appendix 에 있다.

#### 4. Obtaining Model Uncertainty

#### 5. Experiments

- **Model Uncertainty in Regression Task**
- **Model  Uncertainty in Classification Tasks**
- **Predictive Performance**
- **Model Uncertainty in Reinforcement Learning**

#### 6. Conclusions and Future Research



# Reference 

[Dropout as a Bayesian Approximation : Representing Model Uncertainty in Deep Learning - Yarin Gal, Zoubin Ghahramani](https://arxiv.org/abs/1506.02142)





