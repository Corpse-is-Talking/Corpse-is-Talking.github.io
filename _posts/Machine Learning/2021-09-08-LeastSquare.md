---
title: "[ML]5.Linear Least Square and ERM and minibatch Gradient descent"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: Examples for Gradient descent(linear regression) and its ERM approach with minibatch(stochastic) Gradient.
use_math: true
comments: true
---
## 0. Review
지난 포스트에서는 __Gradient Descent__ 와 그 효율을 높여주는 __Backtracking Line Search__ 에 대해서 알아보았다.
강의에서는 가장 흔히사용되는 Least Square Regression 에 Gradient Descent 를 적용하였다. 또, ERM이 어떤식으로 사용되는지를 Gradient, Minibatch, Stochastic Gradient Descent 를 통해 확인해보겠다.

## 1. Setup
[첫번째 포스트](https://lookbackjh.github.io/machinelearning/Introduction-to-Machine-Learning) 에서 이야기한대로,
Linear Least Square Regression 문제의 Input Space, Output Space, Action Space , Loss 를 각각살펴보겠다


- __Input Space__: d개의 feature 을 가진 Input에  관해서 생각할 것이므로, $X= R^d$
- __Output Space__: 결과값 y 는 실수이므로, $Y=R$
- __Action Space__: 위와 마찬가지로, $Y=R$
- __Loss Function__: $L(\hat{Y},Y)=\frac{1}{2}(Y-\hat{Y})^2$
- __Hypothesis Space__: $f$ 는 예측함수, $f$ 로 가능한 함수의 집합을 Linear로 한정시키면,

$$F =\{f:R^d \rightarrow R |f(x)=w^Tx+b, w\in R^d, b \in R\}$$

이제 ERM을 통해서, 가장 이상적인 $f$ ,  즉 coefficient인 $w_1\sim w_d$들을 찾으면 된다.


## 2. ERM 

- __Empirical Risk for given data with size n (To minimize)__

    $$\hat{R}_n(w)=\frac{1}{n} \sum_{i=1}^{n}((w^Tx_i+b)-y_i)^2$$

    우리는 이 $\hat{R}_n(w)$ 를 최소화 하고싶고, 만약 이 함수가, Convex를 만족하고, Differentiable 하다면 , 우리는 Gradient Descent 를 적용 할 수 있다. 


- __Matrix Representation__

    표현의 편의성을 위해 위식을 Matrix형태로 나타내보겠다. 이를위해 X,Y,W 의 차원을 각각설정해주면, 

    $ X: R^{(d+1) \times n}$
    $ w: R^{(d+1) \times 1}$
    $ Y: R^{n \times 1}$
    여기서 $n$은 number of data , $d$는 number of features,$X$와 $w$에서 차원에 있는+1은 일차함수의 상수항을 위해서 표시해두었다. 


   $$\hat{R}_n(w)=\frac{1}{n} \sum_{i=1}^{n}((w^Tx_i+b)-y_i)^2$$
    
    를 Matrix 형태로 다시쓰면 (L-2 Norm)

    $$\hat{R}_n(w)=\frac{1}{n} \parallel X^Tw-Y \parallel ^2 $$

    좀더 알기 쉽게 행렬식을 통해서 표현하면..

$$
    X^Tw-Y =
 \left (\begin{array}{cc}
    1 & x_{1,1} & \cdots & x_{1,d} \\
    1 & x_{2,1} & \cdots & x_{2,d} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
    1 & x_{n,1} & \cdots & x_{n,d}
 \end{array}\right)
\left (\begin{array}{cc}
    w_{0}  \\
    w_{1} \\
  \vdots  \\
    w_{d} 
 \end{array}\right)-
\left (\begin{array}{cc}
    y_{1}  \\
    y_{2} \\
  \vdots  \\
    y_{n} 
 \end{array}\right)
 $$

$$
X^Tw-Y =
\left (\begin{array}{cc}
   w_{0}+w_1x_{1,1}\cdots w_dx_{1,d}-y_1 \\
   w_{0}+w_1x_{2,1}\cdots w_dx_{2,d}-y_2 \\
\vdots   \\
   w_{0}+w_1x_{n,1}\cdots w_dx_{n,d}-y_n
\end{array}\right)
$$








- __Proof of Diffentiability and Convexity__

    1. Diffentiability:

        $\hat{R}_n(w)$은 단순히 미분가능한 1차함수의 합이므로, trivial하게 미분가능하다고 생각할 수 있다. 

    2. Convexity:

        Convex를 증명하는데는 여러가지 방법이 있지만, 
        그중에서 미분가능한 $f$ 에대해서,  $ \nabla ^2 f \geq 0 $ 이면 Convex임을  사용하겠다.
        위에 Matrix Fp








