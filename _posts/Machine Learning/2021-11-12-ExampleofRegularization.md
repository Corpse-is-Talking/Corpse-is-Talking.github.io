---
title: "[ML]9. Implementaion of regularization and Coordinate Descent Method"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: implementing Regularization and Coordinate gradient descent
use_math: true
comments: true
---

### 0. Overview
[저번 포스트](https://lookbackjh.github.io/machinelearning/Regularization/)에서는 Regularization의 두 방법, Coordinate gradient descent 에 대해서 배웠습니다.
이번 포스트에서는 regularization의 구현과, $\lambda$ 값 설정의 방법, 또 더나아가 L1 regularization을 적용한 coordinate gradient descent 를 구현해보겠습니다.

### 1. Regularization 의 구현과 $\lambda$ 찾기

Loss function이 Square loss 일 경우에, Ridge Regression의 공식을 다시한번 살펴봅시다.

  $$ \displaystyle{\min_{w \in R^d}} \sum_{i=1}^{n} (w^Tx_i-y_i)^2+\lambda||w||_2$$

공식에서도 볼 수 있듯이, $||w||_2$ 이들어가서 미분 가능한 형태입니다. $\lambda$ 는 미지수이며, 우리가 찾아야 할 hyperparameter 중하나입니다.
- __Sequence__ 
  그렇다면 가장 적합한 $\lambda$값을 찾기 위해서는 어떻게 해야 할까요? 순서대로 나열해보면,

  1. $\lambda$ 의 범위를 설정하고, 가능한 $\lambda$값들을 만든다.

  2. 각각의 $\lambda$ 에 대하여, train-set 에 적용하여, gradient를 통해 적합한 $w$를 찾는다.

  3. test-set 을 통해, 가장 성능이 좋은 $\lambda$값을 추려낸다.


- __Example__
  그럼 이 과정을 [이전 포스트](https://lookbackjh.github.io/machinelearning/Examples/) 에서 했던 데이터를 토대로 그대로 적용해봅시다. 이 때는 lambda_reg=0 으로 두고 Gradient 를 진행했지만, 이번에는 여러개의 lambda_reg 에 대한 성능이필요합니다. Code는 편의성을 위해, HW에서 제공해준 코드를응용하였습니다(SKlearn의 툴을 이용했습니다).
  


