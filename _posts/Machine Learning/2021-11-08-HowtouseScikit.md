---
title: "[ML]8.How to Use Scikit"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: Using Scikit with python class.
use_math: true
comments: true
---

## 0. Overview
scikit-learn은 데이터를 통해 수학적인 알고리즘을 적용할 때 매우 강력한 python library 입니다. 저도 공부하면서 어떻게 쓰는지 조금씩 알게되었기 때문에, 복습도 할겸 간단하게 이번 포스트와 다음포스트에 걸쳐서 사용법을 작성해보겠습니다. 내용은 scikit-learn 홈페이지[https://scikit-learn.org/stable/developers/develop.html]를 참고했습니다.

## 1. Object of Scikit-learn
Scikit-learn 을 사용하기 위해서는, 클래스를 만들어서 사용하게 되는데(혹은 built-in class), 클래스를 만들때는  주로 4가지를 구현해서 사용하게 됩니다.

#### 1. Estimator
Estimator은 주어진 데이터 X, (y)를 통해 원하는 미지수를 저장합니다. 예를들어, Gradient 를 통해서 w를 구하게 되는 경우, w를 구하는 방법이 fit 함수에 구현되있어야 하고, 특정 데이터의 min or max를 구하는 경우에는 min 혹은 max를 구하는 방법이 구현되어있어야 합니다.
scikit-learn에는 BaseEstimator이 미리 구현되어 있으며, 모든 estimator은 BaseEstimator을 상속받아서 구현되어야 합니다. 모든 estimator은 fit(X,y) 함수를 가지고 있어야 합니다.
BaseEstimator에 대한 자세한 내용은 [여기](https://scikit-learn.org/stable/developers/develop.html)에서 확인 할 수 있습니다.

#### 2. Predictor
Predictor은 fit이 완료된 경우, 구해진 함수가 특정 데이터를 나왔을때, 얼마만큼의 확률로 특정 값을 가질 수 있는지를 보여줍니다. 

#### 3. Transformer
Transformer은 , 데이터를 원하는식으로 조정하는데에 필요합니다. 예를들어 우리가 지금까지 진행해왔던 Regulraization을 하는데에 필요할 겁니다. 모든 transformer 이들어가는 class 는  TransformerMixin를 상속받습니다. 

#### 4. Model
새로운 데이터에 대해, 모델이 얼마나 잘 추정했는가를 점수로 표현해줍니다.

## 2. Examples- MinmaxScaler 구현해보기

Jupyter Notebook을 통해 매우간단하게 1차원 array 에대한 Scikit Minmax scaler을 직접 만들어 봅시다.
- __Code__

  ```python
  import pandas as pd
  import numpy as np
  from sklearn.base import BaseEstimator, TransformerMixin ## inheritance..
  class Minmaxreg(BaseEstimator,TransformerMixin):
      def __init__(self) -> None:
          super().__init__()
      def fit(self,X):
          self.X_max=np.max(X)
          self.X_min=np.min(X)
          return self
      def transform(self,X):
          
          try:
              getattr(self, "X_min") or getattr(self,"X_max") ## checking if fitted
          except AttributeError:
              raise RuntimeError("You must train classifer before predicting data!")
          X=X.copy()
          X=(X-self.X_min)/(self.X_max-self.X_min)
          return X
  ```
  Minmaxreg라는 클래스를 만든 후에, BaseEstimator과 ,TransformerMixin을 상속받아 fit()함수와 transformer()함수를 구현해줍니다.
  저는 fit 함수에 특정 데이터 X의 max와 min을 구할 수 있도록 구현하였고, transform 함수에서는 fit에서 구한 X의 max와 min을 바탕으로, regularization을 시행해줍니다.

- __Test__ 

  간단한 X_train, X_test 데이터를 만들어 실험해봅시다.

  ```python
  X_train=[1,2,3,4,5,6,7,8,9,10]
  X_test=[1,3,5]
  scale=Minmaxreg()
  scale.fit(X_train)
  scale.transform(X_train) ## 결과 1
  scale.fit(X_test)
  scale.transform(X_test) ## 결과 2
  ```
  ![image](https://user-images.githubusercontent.com/75593825/141425552-723248c0-cb27-45fd-8310-84bb01a289ce.png)
  다음과 같은 결과가 나옵니다. 얼핏보면 원하는데로 나온 것 같지만(계산상의 문제는 없습니다) [이전 포스트](https://lookbackjh.github.io/machinelearning/Examples/)에서도 언급했던 대로, test set의 regularization은 train set의 regularization과 같은 기준을 사용해서 진행해야 합니다. 이를 위해서 
  TransformerMixin 는 fit_transform 함수를 제공합니다. 이는 한번 정한 기준을 그대로 고정 시켜주고, 이후에 regularization을 진행할 때, 미리 정한 기준대로 실행합니다. 

  ```python
  print("X_train_reg: {}".format(scale.fit_transform(X_train))) ## 결과 1
  print("X_test_reg: {}".format(scale.transform(X_test))) ## 결과 2
  ```

  ![image](https://user-images.githubusercontent.com/75593825/141426262-40592369-accb-40f1-8518-c843b07345aa.png)

  이제는 X_train의 기준대로, X_test가 regularized 된 것을 볼 수 있습니다.

Scikit-learn 에서 제공하는 MinMaxScaler을 사용하면 훨씬 편리하게 Regularization을 진행할 수 있지만, Scikit-learn에 익숙해지기 위해 간단하게나마, 직접 한번 짜보았습니다
다음포스트에서는 Ridge Regression을 구현, 응용해보겠습니다.

