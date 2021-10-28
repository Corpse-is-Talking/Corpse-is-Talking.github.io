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

그럼이제 Tikhonov Regularization을 이용하여,  L1과 L2 regularization의 차이를 알아보겠습니다. 

- __L1 Regularization(Lasso Regression):__

  ![image](https://user-images.githubusercontent.com/75593825/138661312-0efeb596-6b1a-45e2-901b-07cb3736339a.png)

  
  Loss function으로 Squared loss function을 사용할때, L1 regularizaion을 적용했을때 구해지는 solution입니다.  뒤에 붙는 regularazation term 이 
  $\|w\|$ 이기 때문에 L1이라고 부른다고 합니다.






- __L2 Regularization(Ridge Regression):__


  ![image](https://user-images.githubusercontent.com/75593825/138660644-31b435ea-4179-4461-803c-5e2b195aaa6e.png)

  Loss function으로 Squared loss function을 사용할때, L2 regularizaion을 적용했을때 구해지는 solution입니다. 뒤에 붙는 regularazation term 이 
  $\|w\|^{2}$ 이기 때문에 L2라고 부른다고 합니다.



## 4. Sparsity of Lasso

- __Example__

  ![image](https://user-images.githubusercontent.com/75593825/139176087-d1db6cb2-bef3-4d06-9bbe-57d0e44364b2.png)

  위 그림은 같은 데이터에 대해서 ridge rigression과 Lasso rigression을 동시에 시행하였을 때 나오는 특정 feature의 수렴성을 알아보기 위한 그래프입니다. 
  그래프의 __x축__ 은 feature이 0으로 갈 수록, regularization이 매우 강한걸 나타내고, 1에 근접할 수록, regularization이 매우 적은것(unconstrained erm)을 나타냅니다. 그리고 __y축__ 은 특정 feature들의 현재 feature의 개수에 따른 coefficient값을 나타냅니다.

  그래프에서 확인 할 수 있듯이, Lasso regression은 constraint가 걸려있는상황에서 필요없다고 생각되는 coefficient를 0으로 보내버리는 경향이 매우 강합니다. 그에 반해 Ridge regression은 0 근처값은 주지만, 확실하게 0이라고 하지는 못합니다.



- __Cons of Sparsity__ 

  feature이 많을 경우, iteration을 진행하면 할 수록, overfitting을 방지하기 위해 필요없는 feature의 coefficient를 빠르게 0으로 수렴시키는 것이 좋을때가 있습니다. Lasso regression(L1 reg)는 Ridge regression(L2 reg)에 비해 feature 을 0으로 수렴시키는 경향이 매우 강합니다.(그렇다고 해서 L2 가 L1보다 안좋은 방법이라고는 하지 않습니다!) 이런 경향을 Sparsity라고 부르고 왜 이런 성질을 가지냐에 대해서 간단하게 알아보겠습니다.

- __Reasons of Sparsity__

  그렇다면 Lasso 는 왜 Sparse한 solution을 제공할까요? Cost function에서 둘의 차이는 L1-norm을 쓰냐 L2-norm 을 쓰냐에 있고, lasso 의 sparsity 역시, L1-norm때문에 나옵니다.
  이를 확인하기 위해, 우리의 Decision function을 $f=w_1x_1+w_2x_2$ 라고 해봅시다. 
  밑의 그림은 이 함수에 대한 L2norm 일때의 $w$들의 contour line과 L1norm일때의  $w$들의 contour line을 그린 것입니다.

  ![image](https://user-images.githubusercontent.com/75593825/139178450-22265cda-cbd8-4120-89a0-42e32494ef46.png)

  이상황에서 L1 contour line에 대해 Lasso regularization을 적용한다고 하면 밑의 그림과 같은 상황이 나옵니다.

  ![image](https://user-images.githubusercontent.com/75593825/139178626-7d1479e7-b222-4767-b285-7642dc301919.png)

  $w$를 찾아나가면서, constraint를 만족할때까지 반복을 한 모습을 보여줍니다.(같은 빨간 라인에 있을 경우 같은 Loss 값, 바깥쪽의 빨간 라인이 더 큰 Loss 값)
  
  ![image](https://user-images.githubusercontent.com/75593825/139179546-2d3fde30-78db-4391-807a-23101da140e0.png)

  이 그림은결국 빨간 area나 초록 area에서 constrainted를 만족할 때,  $w_1,w_2$ 둘중 하나는 0이 되므로, Least square을 만족하면서도 sparsity 가 나오기 쉽다는 점을 보여줍니다.
  (Contour line이 원이므로 저런 그림이 형성되고 차원이 더높은 경우에는 원이아니라 타원이 됩니다, 자세한 증명은 넘어가겠습니다.)

  반대로 Ridge의 경우

  ![image](https://user-images.githubusercontent.com/75593825/139179772-9ca23c99-9249-4aed-a575-f23391757883.png)

  사진에서 볼 수 있듯이, feature 둘중 하나가 0이 되기 매우 힘든 구조입니다.


물론 일반적으로 이를 증명하려면 더 복잡한 수식과, 계산이 필요한 것 같습니다. 하지만, 위 그림으로도 충분히 Lasso 가 Ridge에 비해 Sparsity를 가진다는 점을 확인 할 수 있습니다. 그럼 이어서, Lasso Solution을 도출하는 방법에 대해서 알아보겠습니다.


## 5. Deriving Lasso Solution

Gradient Descent를 진행하기 위해서는, 미분이 필요한데, Lasso 를 포함한 Cost function은 절댓값으로 인해, 미분이 불가능합니다. 그렇다면, Lasso의 경우에 어떻게 Gradient Descent를 진행해야 할까요? 








  





