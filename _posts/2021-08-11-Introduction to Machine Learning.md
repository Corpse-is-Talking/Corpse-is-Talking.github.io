---
title: "Introduction to Machine Learning"
categories:
  - Machine Learning 공부
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog

use_math: true
comments: true

---

## 0. Introduction

- 대학원을 준비하면서, OCW를 통해 여기저기서 배운 머신러닝 이론과 
그 응용들을 복습할 겸 블로그를 만들었습니다. 대부분의 이론적인 내용은  [이 강의](https://bloomberg.github.io/foml/#lectures)를 듣고 정리했으며, 추후 올릴 코드들도 위 강의의 숙제를 기반으로 만들었습니다.
- 위 강의는 기본적인 통계지식과 선형대수, 약간의 해석학쪽 지식이 요구되는 것 같습니다. 덕분에 공부할때 고생을 좀 했습니다. 위 강의를 통해 공부하시려는 분들은 참고하시길 바랍니다. 
- 아직 공부중인 단계로, 오류나 잘못된 정보가 작성되어있을 가능성이 있습니다. 이상한 부분을 발견하시면 댓글로 남겨주세요.!! 

## 1. Basic Steps and Action

데이터문제를 마주했을 때 우리는, 다음 세 가지 과정을 통해서 문제를 해결한다.

- 계획을 세운다.
- 계획에 의거해 Action을 취한다.
- 결과물을 도출한다.

여기서 Action을 취한다는 것은 우리가 세운 계획을 통해 무엇이 도출되는가에 대한 이야기다. 예를 몇가지 들어보자.

- 통계시간에 배운 검정에서, 귀무가설 $H_{0} : \theta = 0 $ 을 세우고 기각할지 말지를 결정하는 것
- 머신 러닝 알고리즘을 통해, 특정 물품이 어느 카테고리에 속할지 분류해주는 것
- 잘모르는 것 데이터에 대한 PDF(probability density function)를 도출하는 것.

우리는 최적의 결과를 도출해내기 위해서, 가장 이상적인 Action을 찾을 필요가 있다. 즉, 특정상황에서 SVM, Random-Forest 등등의 다양한 머신 러닝 방법들 중 어떤 머신러닝 방법을 쓸지를 잘 선택할 필요가 있다는 것이다. 



## 2. Formulating Problem

ML문제의 대부분은 다음과 같은 순서로 해결하게 된다.
##### 1. Input data를 관측한다.
##### 2. Action을 취한다.
##### 3. 실제 결과 Y를 확인한다.
##### 4. 함수 $l(A,Y)$ 를 통해 도출된 결과 A(Outcome)와 실제 데이터 Y 의 오차를 수치화한다. ($i.e$, Square loss function)


### Definitions
__Input Space: $X$__ , 주어지는 데이터들의 집합

__Action Space: $A$__ , 수행한 Action을 통해 도출된 결과들의 집합

__Outcome Space: $Y$__ , 결과들의 집합

__Decision Function__ : $x \in X$을 받아서 action $a \in A$를 수행하는 함수 
- $f: X\rightarrow A$ , $x\rightarrow f(x)$

__Loss Function__ : 실제결과값 Y와 action을 통해 도출된 결과 값 을 비교/평가해주는 함수

-  $L$: $A \times Y \rightarrow $ __$R$__ , $(a,y)\rightarrow l(a,y) $

## Example

#### Linear Regression(선형회귀문제):
__d__ 개의 __feature__(특징) 을 가진 __Input X를__  선형회귀를 통해  결과 __Y__ 를 도출한다고 해보자.
여기서 __Input Space__, __Action Space__, __Outcome Space__ 를 각각 정의해보면
- __Input Space: $R^d$__ (feature이 d개이므로 d개의 feature에관한 정보가 모두 필요하다.)
- __Outcome Space: $R$__ (예측하고싶은 실제 값)
- __Action Space: $R$__ (선형회귀 문제는 특정 데이터가 어떤값(R)을 가질 지 예측할때 사용한다)

#### Logistic Regression(로지스틱 회귀문제)
__d__ 개의 __feature__ 을 가진 __Input X를__  로지스틱회귀를 통해  결과 __Y__ 를 도출한다고 해보자.
여기서 __Input Space__, __Action Space__, __Outcome Space__ 를 각각 정의해보면
- __Input Space: $R^d$__ (feature이 d개이므로 d개의 feature에관한 정보가 모두 필요하다.)
- __Outcome Space: 0 or 1__ (Classification 문제에서 주로 사용된다.)
- __Action Space: $R \in (0,1) $__ (로지스틱 회귀의 Action Space는 확률값을 제공해준다 $i.e,$ $P(Y=1)$







