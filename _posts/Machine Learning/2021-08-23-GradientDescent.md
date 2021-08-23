---
title: "[ML]4. Gradient Descent"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: Gradient Descent and Stochastic Gradient Descent
use_math: true
comments: true

---
## 0. Review
지난 포스트까지는 기본적인 Statistical Learning Theory의 Framework에 대해서 알아보았다. 이번 포스트부터는 이 프레임워크들을 이용해서,  ML에서 가장 많이 사용되는 방법 중하나인 Gradient Descent, 더나아가 Stochastic Gradient Descent 까지 알아보겠다.

## 1. Gradient Descent

- #### Definition.
    __Gradient descent__ is a first-order iterative optimization algorithm for finding a __local minimum__ of a differentiable function. (Wikipidea)
    위키피디아의 정의 따르면, __Gradient descent__ 는 미분 가능한 함수의 1차미분계수를 통해, 그 함수의 __local minimum__ 을 찾는 알고리즘이다.


    미분가능한 다변수함수 $F(X)$에 대하여, 어떤 점 X에서,  __Gradient(1차 미분계수)__ __$-\nabla F$__ 는, __현재 위치 X 가 가장 빠르게 감소하는 방향을 나타내준다.__ 이를 이용하여, Step 사이즈를 작게하여, X를 조금씩 움직여서, local minimum 에 도달할때까지 반복하는 것이다. 




    알고리즘을 수학적으로 살펴 보면 다음과 같다.
    #
    For differentiable multi-variable function $F(X)$, very small $\gamma>0$ , with starting point $x_0$, 

    Until Convergence to local minumum, Repeat

    $$ x_{n+1}= x_n-\gamma \nabla F(x_n)$$ ,


    ![image](https://user-images.githubusercontent.com/75593825/130398636-bee6cddf-0d78-4c93-b645-9dc9b6556228.png)
    [Gradient Descent 예시, 출처: 위키피디아](https://en.wikipedia.org/wiki/Gradient_descent){: width="50%" height="50%"}
    그림과같이 Contour Line  (같은 F값을 가지는 X들의 집합 ,바깥쪽 원이 큰 값을 가짐)이 그려져 있고,초기점 $x_0$ 에서 시작하여 Iteration이 진행될수록 local minimum 에 도달해나가는 모습을 볼 수 있다.
- #### Convergence Theroem for fixed step size..
