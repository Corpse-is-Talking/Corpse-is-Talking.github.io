---
title: "[ML]2.Statistical Learning Theory"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: Statistical Learning Theory and Empirical Risk Minimizer..
use_math: true
comments: true

---
## 0. Review
저번 포스트에서는 머신러닝으로 문제를 해결하는 과정과 문제를 설계 할때 사용되는 용어들을 정의했다. 오늘은 이 정의들을 이용해서 일반적인  __Statistical Learning Theory Framework__ 에대해서 알아보겠다.

## 1. Setting
- #### Risk
  데이터$(x,y)$를 독립적으로$(i.i.d)$ 랜덤하게 생성하는 분포 __$ P_{ X\times Y } $__ 가 있다고 하자. 우리는 데이터에대해서 일반적으로 loss function을 작게 해주는 $f$를 구하고 싶을 것이다. 이를 조금더 공식화해보자.
  $P_{ X\times Y }$에서 추출한 데이터 $(x,y)$ 에대하여 decision function $f$를 이용해 구한 loss function의 기대값을 $R$이라고 하면, 



  $$R(f) = E [l(f(x),y)]$$


  라고 쓸 수 있다. 하지만 우리는 $ P_{ X\times Y } $ 가 무엇인지 모르기 때문에 정확한 기대값을 계산할 수 없지만, 예측은 할 수 있다.

- #### Bayes Decison Function.
  __Bayes decision function:__  $$f^*:X \rightarrow A $$ is a function that achieves __minimal risk__ among all possible function.

  즉, Decision function 중에서 리스크를 <span style="color:red">최소화</span>하는 함수를 베이지안 결정함수라고 한다는 것이다.
  이를 수학적으로 나타내보면, Bayes decision function $$f^*$$는


  $$f^*=\underset{f}{\operatorname{argmin}} R(f)$$


  추가로, Decision function이 $f^*$(bayes decision function)일 때의 리스크 $R$을 __Bayes Risk__ 라고 한다.

## 2. Empirical Risk Minimizer.
앞서말했듯이 $P_{X \times Y}$를 모르기 때문에, 정확한 $R(f)$ 를 구할 수 없다. 하지만, 주어진 데이터가 많은경우 우리는 이 리스크를 예측 할 수 있게된다.

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



