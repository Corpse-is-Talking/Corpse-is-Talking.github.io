---
title: "[ML]8. Implementaion of regularization and Coordinate Descent Method"
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

공식에서도 볼 수 있듯이, 