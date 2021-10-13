---
title: "[ML]7. L1 and L2 regularization과 비교"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: L1 and L2 regularization and its comparison
use_math: true
comments: true
---

## 1. Overview
[바로 이전 포스트](https://lookbackjh.github.io/machinelearning/Examples/)에서는 기본적인 Gradient descent 및 Stochastic gradient를 구현했다.  Regularziation parameter을 포함해서 구현했지만, 추후에 다루기 위하여 0으로 두고 결과를 보았었다. 하지만, 실제 데이터를  Regualarization parameter은 중요한 hyperparameter 중하나라고 한다. 이번포스트에서는 regularzation의 기능과 필요성 그리고 종류들에 대헤 알아볼 것이다. 또, 저번 포스트에서는 단순히 Train-loss를 줄이는 방법만 사용했지만, 이번 포스트에서는  Test-loss 를 줄이기위한 과정, Machine learning 문제의 진행과정에 대해 더 자세히 알아보겠다. 


## 2. Regularization이 필요한 이유

![image](https://user-images.githubusercontent.com/75593825/137084674-51c438d7-d335-4a0b-94bb-24273876f241.png)
(출처: Andrew Ng 교수님의 Coursera 강의)
그림처럼 데이터가 주어지고, feature이 많을 때, 과적합하게(모든 feature 에대한 coefficients 가 0이 아닌 값을 가지게 되는 것)되는 경우가 있다. 이를 Overfitting이라고 하고, Overfitting이 발생할 경우 Train Set의 Error가 낮게 나왔음에도, Test Set 에 대한 Error은 높게 형성되는 경우가 자주나온다. (Train error은 작지만 Test error이 크다는 것은 실제 데이터에 대한 예측력이 부족함을 나타낸다). 이런 Overfitting을 줄여주기 위해 도입되는 것이 Regularization이다. Regularization을 진행하는 방식은 매우 간단하다. Loss function에다가 $\frac 12\lambda^2 w$(L2-Regularization일 경우)를 더해주면 된다. 더하게 된다면, 우리는 Loss function을 최소화 하고싶기 때문에, 몇몇의 coefficients들은 0이 될것이다.(0이 된다는 것은, 모든 feature의 coefficients가 0보다 크게나와 overfitting이 일어나는 것을 방지한다는 것이다.) Overfitting의 종류에는 대표적으로 L1-Regularization과 L2-Regularization이 있고, 이 두가지가 가장 많이 쓰인다고 한다. 

