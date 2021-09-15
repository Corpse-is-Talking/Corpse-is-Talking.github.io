---
title: "[ML]5.ERM of Linear Least Square  and minibatch Gradient descent"
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

    이해하기 쉽게  행렬식을 풀어서 써보면..

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

$$\hat{R}_n(w)=\frac{1}{n} \parallel X^Tw-Y \parallel ^2 = \frac{1}{n} \sum_{i=1}^{n}(w_0+w_1x_{i,1}+\cdots w_dx_{i,d}-y_i)^2$$



- __Proof of Diffentiability and Convexity__

    1. __Diffentiability__:

        $\hat{R}_n(w)$은 단순히 w에 대해서 미분가능한 1차함수의 제곱의 합이므로, trivial하게 미분가능하다고 생각할 수 있다. 

    2. __Convexity__:

        Convex를 증명하는데는 여러가지 방법이 있지만, 
        그중에서 미분가능한 $f$ 에대해서,  $ \nabla ^2 f \geq 0 $ 이면 Convex임을  사용하겠다.

        $$
        \nabla R_n =
        \left (\begin{array}{cc}
         {\partial\over\partial w_1}R_n \\
          {\partial\over\partial w_2}R_n \\
        \vdots   \\
          {\partial\over\partial w_d}R_n
        \end{array}\right)=
        2\times X(X^Tw-Y)= 2\times X
        \left (\begin{array}{cc}
          w_{0}+w_1x_{1,1}\cdots w_dx_{1,d}-y_1 \\
          w_{0}+w_1x_{2,1}\cdots w_dx_{2,d}-y_2 \\
        \vdots   \\
          w_{0}+w_1x_{n,1}\cdots w_dx_{n,d}-y_n
        \end{array}\right)
        $$

        $$\nabla^2 R_n =
        \left (\begin{array}{cc}
         {\partial^2\over\partial w_0}R_n \\
          {\partial^2\over\partial w_1}R_n \\
        \vdots   \\
          {\partial^2\over\partial w_d}R_n
        \end{array}\right)=2
        \left (\begin{array}{cc}
        \sum_{i=1}^{n}1^2 \\
        \sum_{i=1}^{n}x_{i,1}^2 \\
        \vdots \\
        \sum_{i=1}^{n}x_{i,d}^2
        \end{array}\right) \geq0
        $$

        따라서, Empirical Risk는 Convex하고, Gradient Descent 를 적용 할 수 있다..
        
## 3. MiniBatch Gradient Descent

- __Definition__
  일반적인 __Gradient Descent__ 를 실행 할 때에는 한개의 데이터 셋이 들어올때마다, Loss를 구하고, 업데이트 해준뒤 다음 Step으로 넘어가게 된다. 하지만 역시 데이터 전체를 건드려야하기 때문에 시간소요가 크다. 이를 보완하기 위해서, Loss를 전체 데이터에 대해서 구하지 않고, 전체 데이터를 여러개의 __minibatch__ 로 나누어서 구하는 방법을 __minibatch Gradient Descent__ 라고 한다. 이 방법을 사용하면, 수렴속도가 훨씬 빨라져, practical 한상황에서는 대부분 minibatch(about size=1~32)를 사용하고, 이 때 mini-batch 의 size가 1이라면, __stochastic gradient descent__ 라고 한다.


- __Unbiasedness of Minibatch gradient descent__
  통계시간에 배웠듯이, 모수 $\theta$를 추정한다고 할때, 가능하다면, __unbiased__ 를 만족하는 $\hat{\theta}$, 즉, $E[\hat{\theta}]=\theta$ 를 사용할 것이다.
  Mini-batch를 사용할때, minibatch를 이용해서 구한 Loss function에 대한 gradient 값이 전체 데이터의 Loss function에 대한 gradient의 unbiased 한 estimate이 될 수 있는지 알아보자. 
  
  
  원래 n개의 데이터에서 구한 gradient 는
  $$ \hat{R}_n(w)= \frac{1}{n} \sum_{i=1}^{n} \nabla _w l(f_w(x_{i},y_{i})) $$


  여기서 n개의 데이터를 사이즈가 N인 batch M개로 만든다고 생각해보자
  m번째 batch에 있는 데이터는
  
  $$ (x_{m1},y_{m1}),\dots(x_{mN},y_{mN})$$

  이 minibiatch를 이용해 구한 gradient는

  $$ \hat{R}_N(w)= \frac{1}{N} \sum_{i=1}^{N} \nabla _w l(f_w(x_{mi},y_{mi})) $$

  이제 unbiased의 여부를 확인해보기위해서, 기댓값을 구해보면,

  $$
  \begin{align}
  E[\hat{R}_N(w)]&= 
  \frac{1}{N} \sum_{i=1}^{N} E[\nabla _w l(f_w(x_{mi},y_{mi}))]\\
  &=  E[\nabla _w l(f_w(x_{m1},y_{m1}))]\\
  &= \sum_{i=1}^n P(m_1=i)\nabla _w l(f_w(x_{i},y_{i}))\\
  &=\frac{1}{n} \sum_{i=1}^n\nabla _w l(f_w(x_{i},y_{i})\\
  &=\nabla \hat{R_n(w)}
  \end{align}
  $$

  여기서 전제사항은, minibatch는 random하게 같은 확률로 선택된다는 것이다.(따라서 (1)~(4)의 전개가 가능하다)
  즉 minibatch의 의 gradient는 full batch의 gradient의 __unbiased estimate__ 이 될 수가 있고, 이를 이용해 결과를 도출해내면 되는것이다.

- __Efficieny__:
  
  Batch size N에 대하여

  __N이 크다면__ , gradient의 예측값은 매우 좋지만, full batch와 같은 이유로, 수렴이 느려질 수 있다

  __N이 작다면__ , gradient의 예측값은 불안정하지만, 데이터를 조금만 거쳐도 되기때문에, 수렴이 훨씬 빠르다.


  __Recomendation :__ practical 한상황에서 N은 hyper parameter의 일종이며, 1~몇백 사이의 숫자가 가능하다.
  Deep learning의 대가 중 한분인 Youshua Bengio 교수님은 약 N=32가 좋은 설정값이라고 하셨다고 한다.


  다음포스트는 코드와 함께 Gradient, minibatch gradient, stochastic gradient의 차이를 알아보겠다.


## 4. 참고문헌

  [Introduction to Statistical Learning Theory, Sgd lecture note](https://davidrosenberg.github.io/mlcourse/Archive/2017Fall/Lectures/02b.SGD.pdf)








