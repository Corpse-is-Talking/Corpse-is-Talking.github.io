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


  라고 쓸 수 있다. 하지만 우리는 $ P_{ X\times Y } $ 가 무엇인지 모르기 때문에 정확한 기대값을 계산할 수 없다. 하지만, 우리는 통계적인 방법을 이용해서 기댓값을 예측을 할수 있게 된다.

- #### Bayes Decison Function.
  __Bayes decision function:__  $$f^*:X \rightarrow A $$ is a function that achieves __minimal risk__ among all possible function.

  즉, Decision function 중에서 리스크를 <span style="color:red">최소화</span>하는 함수를 베이지안 결정함수라고 한다는 것이다.
  이를 수학적으로 나타내보면, Bayes decision function $$f^*$$는


  $$f^*=\underset{f}{\operatorname{argmin}} R(f)$$


  추가로, Decision function이 $f^*$(bayes decision function)일 때의 리스크 $R$을 __Bayes Risk__ 라고 한다.

## 2. Decision Theory
- #### Setting
  Bayes decision function을 결정하는 과정을 살펴보자.
  데이터 x가 주어졌을때, y를 예측하는 함수가 $f$ 가 있다고 하자. 우리는 일반적으로 작은 리스크를 선호할 것이며, 리스크를 최소화하는 함수를 Bayesian Decision Function이라고 정의했다. 이를 수학적으로 나타내보자.
  데이터쌍 $(X,Y)$가 분포 $P(X,Y)$로부터 주어졌다고 하자, 기댓값의 정의로부터

  $$ r(f):= \int\int L(f(X),Y)p(X,Y)dX\ dY $$
  
  즉 $f$ 에관한 함수 $f$ 를 최소화하는 문제이다. 이어서 optimal $f$를결정하는 과정을 알아보자.

- #### Optimal Decision Rule
  Optimal Decision Rule: For each $X$ choose the prediction $f(X)$ that minimizes the conditional expected risk
  즉, 특정 데이터 쌍 $(X,Y)$가 주어졌을때, 주어진 $X$ 에관한 conditional risk 를 최소화하는 $f$가 optimal 이라는 것이다.
  이를 수학적으로 표현해보면, 

  $$ f_{opt}= argmin_f \int L(f,Y)p(Y|X)dY $$

  여기서 적분이 한번만 되는 이유는 한 데이터 쌍에 대해서만 구하기 때문이고, Conditional Risk 를 최소화해야하기 때문에, Conditional Probability 가 주어졌다.
  이런식으로 모든 데이터쌍에 대해, 각각의 $f_{opt}$ 를 구해주면, 우리가 원하는 Expected Risk가 최소화된다는것은 간단하게 증명 할 수 있다.
  통계시간에 배우는 Bayesian 정리 로부터 $P(X,Y)=P(Y|X)P(X)$로 바꿔서 쓸 수 있고, 어떤 함수 $f$에 대해서, 

  $$ r(f):= \int(\int L(f(X),Y)p(Y|X)dY)P(X)dX $$

  로 쓸 수 있게 된다. 아까 정한 함수 $f_{opt}$ 에대하여, 가운데 괄호친 부분은 최소화 되고
  모든 함수 $f$에 대하여, 

  $$ r(f) \geq \int(\int L(f_{opt}(X),Y)p(Y|X)dY)P(X)dX $$

  따라서 특정 데이터 $X$ 가주어졌을때 그 conditional risk 를 최소화하는 함수를 Optimal Decision Function혹은 Bayesian Decision function이라고 할 수 있다.






## 3. Examples
- #### 0-1 Loss
  분포 $P(X,Y)$ 가 주어지고 Y는 k개의  Discreteg한 category 로 분류된다고 하고($Y=y_{1},...,y_{k}$), Loss Function $l$은 
  
  $$l(a,Y)= 1(a\neq Y)$$
  으로 정의된다고 하자. (0 if a=Y else 1)
  이 경우에 Optimal Decision Rule 을 결정해보자.  데이터 $x$ 가 주어진다면, 그의 결과 $y$는 분포 $y|x$ 로부터 주어지고, 0-1 loss 의 Conditional Risk를 수식으로 표현해보면, i번째 카테고리가 결과값이라고 했을때

  $$
  \begin{aligned}
  r(f|x)&= \sum_{i=1}^{k} 1(f(x)\neq y_i)P(y_i|x) \\
  &=  1-1(f(x)=y_i)P(y_i|x)
  \end{aligned}
  $$
  
  이다. 결국 이 문제는
  $P(f(x) \neq Y)$ 를 최소화하는 문제로 바뀌고, 이는 다시쓰면 각각의 category에 대하여

  $$ P(f(x) = Y)$$
  
  를 최대화 하는 문제가 되고, decision rule 은 , 주어진 x에 대해서 가능한 y값들($y_1\ to \ y_k$ ) 중 가장 확률이 높은걸 고르는 것이 될 것이다.
  이를 수식으로 표현하면,

  $$ f^*=argmax_{y}p(Y=y|X=x)  $$




- #### Least Square Regression.
  Loss function $l(a-y)=(a-y)^2$ 으로 정의된다고 하자. 
  이 경우에 Optimal Decision Rule을 결정해보자.
  데이터 $x$ 가 주어진다면, 그의 결과 $y$는 분포 $y|x$ 로부터 주어지고, Squared loss 의 Conditional Risk를 수식으로 표현해보면,  

  $$
  \begin{aligned}
  E[(f(x)-y)^2|x]&=E[(f(x)-E[y|x]+E[y|x]-y)^2|x] \\
  &=E[(f(x)-E[y|x])^2|x]+E[(E[y|x]-y)^2|x]+2E[(f(x)-E[y|x])(E[y|x]-y)|x] \\
  \end{aligned}
  $$



  증명을 위해서 햇갈릴 만한 개념 세가지만 짚고 넘어가자  
  $$
  \begin{aligned}
  1.&E[y|x]는 ,\ x에관한 \ 함수이다  \\
  2.&E[E[y|x]]=E[y]  이다. (Law \ of \ Iterated \ Expectaion.)\\
  3.&E[g(x)Y|X=x]=g(x) \times E[Y|X=x](g(x) 를 \ 상수취급 \ 가능 )
  \end{aligned}
  $$


  잘 와닿지 않는다면, [여기](https://www.youtube.com/watch?v=yDkm9AYaczk)를 참고해보자.
  

  $g(x)=E[y|x]$   
  라고 하자. 문제를 다시써보면

  $$
  \begin{aligned}
  E[(f(x)-y)^2|x]&=E[(f(x)-E[y|x]+E[y|x]-y)^2|x] \\
  &=E[(f(x)-g(x))^2|x]+\textcolor{blue}{E[(g(x)-y)^2|x]}+\textcolor{red}{2E[(f(x)-g(x))(g(x)-y)|x]} \\
  \end{aligned}
  $$

  여기서 빨갛게 칠한 부분을 보자. $(f(x)-g(x))$ 는 위에서 말했듯이 상수취급해줄 수 있고,
  그렇게 된다면 남는건, $E[(g(x)-y)|x]$인데, 이를 정리하면,
  
  $$
  \begin{aligned}
  E[(g(x)-y)|x]&=E[g(x)|x]-E[y|x]\\
  &=g(x)-E[y|x] \\
  &= E[y|x]-E[y|x](앞서서, g(x)=E[y|x]로 정의했다.) \\
  &=0
  \end{aligned}
  $$

  따라서, 빨갛게 칠한 부분은 0이되고, 
  $$E[(f(x)-y)^2|x]=E[(f(x)-g(x))^2|x]+ \textcolor {blue}{E[(g(x)-y)^2|x]}$$ 
  이렇게 쓸 수 있다.  

  다시 위에수식으로 돌아가 이번엔 파란글씨부분을 살펴보자, 
  우리가 최소화하고자하는 함수 $E[(f(x)-y)^2|x]$는 $f$에 관한 함수이며,  $E[(g(x)-y)^2|x]$ 는 $f$에관한 항이 하나도 없으므로, 상수로 생각할 수 있다.  결국 $E[(f(x)-y)^2|x]$ 를 최소화하기 위해서는, $E[(f(x)-g(x))^2|x]$ 을 최소화 하는것이랑 같고,이를 최소화하기 위해서 $ f=g(x) $, 이어야하고, $f ^*=E[y|x ]$ 이된다.

  여기까지, 우리는 주어진 데이터 $ x $ 에 대하여  conditional risk를 최소화해주는 $ f^* $ 을 구했다. 
  모든 데이터 x에 대해서, 각각의 Conditional Risk 를 최소화시켜주는 $f^*$가 존재할 것이며,   

  $$ E[(f^*(x)-y)^2|x] <=E[(f(x)-y)^2|x] $$

  가 성립하고, 

  $$E[(f^* (x)-y)^2]=E[E[(f^* (x)-y)^2|x]]$$  
  
  이므로(Law of Iterated Expectation),


  $$ E[(f(x)-y)^2]=\int\limits E[(f(x)-y)^2|x] P(X=x)dx\geq\int\limits E[(f^* (x)-y)^2|x] P(X=x)dx$$


  따라서,주어진 데이터 $x$에 대하여 Optimal Decision Rule은 
  $f^*=E[y|x]$ 이다.
  사실, Expected Risk를 적분형태로 표현해서, 미분을통해 구하는 방법이 훨씬 간단한지만, 강의에서는 이 방법을 사용했다.






## 3. Empirical Risk Minimizer.
하지만 우리는  $P_{X \times Y}$를 모르기 때문에, 정확한 $R(f)$ 를 구할 수 없다. 하지만, 주어진 데이터가 많은경우 우리는 이 리스크를 예측 할 수 있게된다.

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
  






