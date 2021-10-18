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
[바로 이전 포스트](https://lookbackjh.github.io/machinelearning/Examples/)에서는 기본적인 Gradient descent 및 Stochastic gradient를 구현했습니다.  Regularziation parameter을 포함해서 구현했지만, 추후에 다루기 위하여 0으로 두고 결과를 보았었습니다. 하지만, 실제 데이터를  Regualarization parameter은 중요한 hyperparameter 중하나라고 합니다. 이번포스트에서는 regularzation의 기능과 필요성 그리고 종류들에 대헤 알아볼 것입니다. 또, 저번 포스트에서는 단순히 Train-loss를 줄이는 방법만 사용했지만, 이번 포스트에서는  Test-loss 를 줄이기위한 과정, Machine learning 문제의 진행과정에 대해 더 자세히 알아보겠습니다. 


## 2. Regularization이 필요한 이유

![image](https://user-images.githubusercontent.com/75593825/137084674-51c438d7-d335-4a0b-94bb-24273876f241.png)



(출처: Andrew Ng 교수님의 Coursera 강의)


그림처럼 데이터가 주어지고, feature이 많을 때, 과적합하게(대부분의 feature 에대한 coefficients 가 0이 아닌 값을 가지게 되는 것)되는 경우가 있습니다.

 이를 Overfitting이라고 하고, Overfitting이 발생할 경우 Train Set의 Error가 낮게 나왔음에도, Test Set 에 대한 Error은 높게 형성되는 경우가 자주나옵니다. (Train error은 작지만 Test error이 크다는 것은 실제 데이터에 대한 예측력이 부족함을 나타냅니다). 
 
 
 ## 3. Regularization
 이런 Overfitting을 줄여주기 위해 도입되는 것이 Regularization이라고 합니다. Regularization을 걸어주는 방법에는 제약식을 걸어주는 Ivanov의 방법과, loss function에 페널티를 더해주는 Tikhonov방법이 있다고 합니다.. 

 - __Ivanov Regularization:__
함수 f가 있고 그 함수의 복잡도를 일정수준이하로 제약을 걸어주는 것입니다.
(예를들어 f가 다항식이면 f의 최고차항계수를 r이하로 맞추는것)

![image](https://user-images.githubusercontent.com/75593825/137688117-ff5671e6-2b1f-4cf0-94f5-8f844bb680ad.png)

- __Tikhonov Regularization:__
함수가 특정 차수를 가지게 될경우 그경우의 복잡도와 $\lambda$를 곱해 페널티를 부여하는 것이다.

![image](https://user-images.githubusercontent.com/75593825/137688287-680f784b-7273-4de4-9c44-5450461286b1.png)

대부분의 경우 이바노프와 티코노프의 방법의 결과값은 같게 도출된다고 하고, (이에대한 자세한 증명은 생략하겠습니다). 티코노브의 방법이 Unconstrained 되어있기 때문에 선호된다고 합니다. 

그럼이제 Tikhonov Regularization을 이용하여, L1과 L2 regularization의 차이를 알아보겠습니다. 

- __L1 Regularization(Lasso Regression):__



- __L2 Regularization(Ridge Regression):__

