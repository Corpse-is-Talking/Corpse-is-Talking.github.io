---
title: "[ML]3. Empirical Risk Minimizer and Hypothesis Space"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: Introduction of Empirical Risk, Hypothesis Space..
use_math: true
comments: true

---
## 0. Review
지난포스트에서는 Risk, Bayesian Risk, Bayesian Decison function의 정의와, 그 적용 사례들을 살펴보았다. (0-1 loss and Squared-loss)
하지만 지난 포스트에서 진행된 모든 사례들은 $P_{X \times Y}$ 가 어떤 분포인지 안다는 가정하에 진행하였다. 실제로는 이 $P_{X \times Y}$ 를 알 방법이 없기에, 우리는 지난포스트에서 구한 대로, Bayesian Decision function을 구할 수 없게된다. 이런상황에선 어떻게 해야할까? 이를 해결하기위해 Empirical Risk Minimizer 개념이 도입되었다고한다. 


## 1. Empirical Risk Minimizer.
우리는  $P_{X \times Y}$를 모르기 때문에, 정확한 $R(f)$ 를 구할 수 없다. 하지만, 주어진 데이터가 많은경우 우리는 이 리스크를 예측 할 수 있게된다.

- ####  Strong Law of Large number.
  기대값이 $m$인 분포 $z$ 로부터 $i.i.d$하게 $z_1, z_2,...z_n$ 을 추출했다고 하면, 


  $$\underset{n\rightarrow \infty}{\operatorname{lim}} \frac{1}{n}\sum_{n=1}^{\infty} z_i = m $$


  을 1의 확률로(Almost Surely)만족하고, 이를 큰 수 의 법칙이라고 한다.
  즉, 특정 분포를 가진 모집단으로부터 데이터를 뽑는 경우에, 데이터를 많이 뽑을수록  표본평균은 모집단의 평균에 수렴한다는 것이다.
  증명은 [여기](https://www.youtube.com/watch?v=Yh5bR7X3ch8) 를 참고하면 좋을 것같다. (사실 영상은 Weak law of Large number에 대한 증명이지만, insight를 얻기에는 충분하다고 생각한다.. Strong Law of Large number의 명확한 증명은 필자의 수학실력으로는 아직 이해하기 무리인 것 같다.)

- #### Empirical Risk
  $P_{X \times Y}$ 에서 뽑은 $i.i.d$ 한 $n$개의 데이터셋을 $D_n =((x_1,y_1),...,(x_n,y_n))$이라고 하자.Decision function $f$에 대하여 뽑은 $n$개의 데이터셋의 평균을 $R_n(f)$이라고 하면,


  $$R_n(f)=\frac{1}{n}\sum_{i=1}^{n} l(f(x_i,y)) $$


  여기서 큰수의 법칙을 적용해보면 

  
  $$\underset{n\rightarrow \infty}{\operatorname{lim}} R_n(f)=R(f)$$ 
  
  즉, 데이터수 $n$이 많을 수록 $R_n(f)$ 는 $P_{X \times Y}$의 실제 리스크 $R(f)$로 수렴하게 되고, 이 때의 $R_n(f)$를 __Empirical Risk__ 라고 한다. 
  __요약하자면, 우리는 데이터 수가 많을 때, $R_n(f)$를  $R(f)$로 생각해도 되고, 실제 $P$가 어떤 분포인지 몰라도 $R(f)$를 구할  수 있다는 것이다..__

- #### Empirical Risk Minimizer

  앞서구한 Empirical Risk를 최소화해주는 함수 $f^*$을 Empirical Risk Minimizer 이라고 하고, 이를 수학적으로 표현하면


  $$ f^*=\underset{f}{\operatorname{argmin}} \, R_n(f)$$
  

  즉, 주어진 데이터의 risk를 최소화 하는 Decision Function을 구하면 된다.

## 2. Hypothesis Space.

- #### Problem
    ERM을 구할 때,  Decision function의 범위를 지정해두지 않으면, Optiaml Decision function은 하나가 아닐 수 도 있으며, Overfitting 등 다양한 문제가 발생할 수 있다. 
    강의에서 사용한 예로부터 어떤 문제가 발생 할 수 있는 지 확인해보자.
    ![image](https://user-images.githubusercontent.com/75593825/130394998-8c996c28-6d7b-438b-a9c2-aa98d7a08a91.png)

    실제 $P_{X \times Y}$ 를 알고 있는 상황속에서 ERM을 구해보는 과정이다, $X$ 는 [0,1]에서 Uniform Distribution을 따르고, Y는 항상 1인 분포가있고, ERM을 위해 그중에서 데이터 3개를 추출했다고 하자, 
    ![image](https://user-images.githubusercontent.com/75593825/130395271-7b111cf3-dea7-4d5a-8070-11d80389c946.png)
    여러개의 Empricial Risk 를 최소화하는 Decision Function이 존재 할 수 있고, 그중에 그림에서 제안한 함수같은 Decision Function을 고안했다고 해보자, 이 경우 Empirical Risk 는 최소화시키는데에 성공했지만, 실제 분포를 기준으로 보았을때, 오류투성이인 Decision Function이다.(x가 0.25,5, 0.75 가 아닐 경우에는 전부 틀린 예측값을 전달해준다.) 물론 데이터 수가 극단적으로 적은 상황이긴 하지만, 실제로 데이터가 많은 상황에서도 ERM에는 성과가 좋지만, 실제 데이터에는 성과가 나쁜, 위와같은 상황이 충분히 벌어질 수 있다고 한다. 


- #### Constrained ERM
    위에서 말한 Decision Function의 범위를 제약시키지 않을 때 생기는 문제를 해결하기 위해서 , $f$로 가능한 범위를 제약시켜서 문제를 푸는것을 Constrained Empirical Minimization 이라고 한다. 
    (Ex, 오직 다항식 형태의 함수만 사용한다던지, Continuous 한 function만 사용한다던지 등등)

    Hypothesis Space, 즉 가능한 함수들의 집합을 $F$라고 할때, 
    Cosntrained ERM을 수학적으로 표현해보면, 

    $$ \hat{f_n} =\underset{f\in F}{\operatorname{argmax}} \  \frac{1}{n} \sum_{i=1}^{n}l(f(x_i),y_i) $$



## 3. 참고 문서
[Introduction to Statistical Learning theory Lecture Slide](https://davidrosenberg.github.io/mlcourse/Archive/2017Fall/Lectures/02a.intro-stat-learning-theory.pdf)


