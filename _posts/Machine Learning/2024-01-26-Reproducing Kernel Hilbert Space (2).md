---
title: "[ML]9.Reproducing kernel Hilbert Space (1)"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: RKHS(2), Definition of RKHS.
use_math: true
comments: true
---
# Reproducing Kernel Hilbert Space (2)

저번 포스트에서는 RKHS를 정의하기 위한 기본적인 개념(공간에 대한 정의부터 Riesz Representation Therorem까지) 에대해 짚었습니다. 이번 포스트에서는 이 개념들을 잘 활용하여 Reproducing Kernel hilbert space에 대해 설명하고, 그 개념들을 몇가지 짚어보려고 합니다.

## Reproducing Kernel Hilbert Space

*Notation적으로 $f(\cdot)$은 any $x \in X$를 받는 함수를 가리킵니다.  

$H$는 이제 $X$를 domain으로 받는 함수들의 집합을 나타내며, for 모든 $x \in X$ 에 대해서, mapping 해주는 특별한 functional 이 $H$ 에 존재한다. 

여기서 햇갈릴만한 부분이 있다면, function들의 집합 또한 vectorspace(사실 vectorspace이므로, function=vector이다.)를 구성할 수 있고 구성된 vector space에서 proper한 inner product를 정의 할 수 있다면, function들로 이루어진 Hilbert Space를 구성 할 수 있다.  

여기서 이전 포스트에서 다뤘던 Riesz Representation Theorem 에 대해서 다시 Remind해보겠습니다.

“Evaluation itself can be represented as an inner product. “ 즉 f에서의 evaluation은 H 의 어떤 함수(벡터)와의 inner product로 나타낼수 있다”

### Evaluation Functional

Let $H$ be a Hilbert space of functions $f:X\rightarrow \R$ , defined on a non-empty set $X$ . For a fixed $x \in X$,  map $\delta_x:H\rightarrow \R,$  $\delta_x:f\rightarrow f(x)$ is called evaluation functionals

$X$의 원소들을 인자로 받는 function들로 구성된 Hilbert Space $H$ 에서 $f$에 $x$를 대입하는 행위를 evaluatioin functional이라고 한다. (이는 riesz representation theorem 에 의해서 다른 적절한 function과 내적하는 행위와도 같다. ) 

### Reproducing Kernel Hilbert Space

A Hilbert space $H$ of functions $f: X \rightarrow \R$, defined on a non-empty set $X$ is said to be a reproducing kernel hilbert space if $\delta_x$ is continuous $\forall x \in X$

사실 정의만 읽어보면 도대체 뭘 말하는건지 모르겠습니다.

일단은 Evaulation functionals $\delta_x$가 모든 $x$ 에 대해서 continuous한 공간을 RKHS라고 생각해둡시다.

### Reproducing Kernels

들어가기 전에, kernel도 function이다 라는 것을 항상 생각하면서 보면 도움이 됐던 것 같습니다.

A function $k:X\times X \rightarrow \R$ is called a reproducing kernel iff  it satisfies 

- $\forall x \in X, k(\cdot,x) \in H$
- $\forall x \in X, \forall f  \in H, <f,k(\cdot,x)>_H=f(x)$

In particular , for any $x,y \in X$,

$k(x,y)=<k(\cdot,x), k(\cdot,y)>_H$

즉 $f$의 evaulation functional과 같은 역할을 하는 kernel이 힐베르트 공간 $H$에 항상 Unique하게  존재한다는 것입니다.

먼저 Reproducing Kernel의 Uniqueness 부터 증명해보자

**If reproducing kernels exists in hilbert space, it is unique.** 

pf)

Assume $H$ has two reproducing kernels $k_1$  and $k_2$. Then, 

$$
<f,k_1(\cdot,x)-k_2(\cdot,x)>_H=f(x)-f(x)=0 \ \forall f \in H , \forall x \in X
$$

taking $f=k_1(\cdot,x)-k_2(\cdot,x)$, we obtain $||k_1(\cdot,x)-k_2(\cdot,x)||^2_{H}=0$

which in turn tells that $k_1=k_2$ and reproducing kernels are unique. 

**즉, $f$의 evaluation이 가능하게 하는 kernel이 힐베르트 공간 $H$에 unique하게 존재한다는 것입니다. (Riesze Representation Theorem에 의해 어떤 함수 g가 힐베르트 공간 $H$에 존재하는데 그 g가 unique하다는 것입니다.)**

$**H$ is a reproducing kernel hilbert space if and only if $H$  has a reproducing kernel**

pf) H has reproducing kernel→ H is RKHS

Given that a Hilbert Space $H$ has a reproducing kernel $k$ with the reproducing prorperty 

 $<f,k(\cdot,x)>_H=f(x)$, then 

$$
\begin{align} 
	|\delta_x f| & =|f(x)|\\ 
    	& =|<f,k(\cdot,x)>_H|\\ 
& \leq||k(\cdot,x)||_H||f||_H\\ 
        & =|<k(\cdot,x),k(\cdot,x)>_H|^{0.5}
||f||_H\\
&=k(x,x)^{0.5}||f||_H\end{align}
$$

여기서(2)는 Riesz Representation Therorem,  (3)는Cauchy scwarz inequality에 의해서 성립하고(Vector space에서의 ),  (4)→(5)는 위에kernel의 특성을 참고. 

결론적으로 $\delta_x$는 bounded linear operator이 되므로, $\delta_x$가 continous 하므로, H는 RKHS가 된다. 

반대방향으로의 증명: H is RKHS→ H has a reproducing kernel

By Riesz Representation Theorem, in Hilbert space, there exist an element s.t $\delta_xf=<f,f_{\delta_x} >$

Define $k(x',x)=f_{\delta_x}(x'), \forall x,x' \in X$. Then, clearly, $k(\cdot,x)=f_{\delta_x} \in H$ and $<f,k(\cdot,x)>_H=\delta_xf=f(x)$. Thus, $k$ is the reproducing kernel . 

### **Kernel**

**Definition**

Let $X$ be a non-empty set. The function $k: X\times X \rightarrow \mathbb{R}$ is said to be kernel if there exitst a real Hilbert space  $H$ and a map $\phi:X\rightarrow H$ such that $\forall x,y \in H,$

$$
k(x,y)=<\phi(x),\phi(y)>_H
$$

여기서 $X$를 $H$의 공간으로 mapping 해주는 $\phi$를  feature mapping function이라고 하고, 유일하지는 않다. (kernel은 Unique하지만, Featuremapping은 Unique하지 않습니다.,)

 

**Example**

Consider $k(x,y)=xy=[\frac{x}{\sqrt{2}}\frac{x}{\sqrt{2}}][\frac{y}{\sqrt{2}}\frac{y}{\sqrt{2}}]^T$

where we can define feature mapping as, $\phi(x)=x$  or $\phi(x)=[\frac{x}{\sqrt{2}}\frac{x}{\sqrt{2}}]$

이 예제에서 확인 할 수 있는 점은, kernel의 결과가 나오게 하는 feature map이 하나 이상 존재 할 수 있다는 것이다. 

**Corollary**

In Hilbert space with reproducing kernel  $k$ , take $\phi(x):x \rightarrow k(\cdot,x)$ 

**Insight**

여기서 얻을 수 있는 일종의 insight는 

**Positive Definete Functions(kernels)**

- A symmetric Function $h: X\times X \rightarrow \R$  is positive definite if $\forall n \geq1, \forall(a_1,\cdots,a_n) \in \R^n, \forall(x_1,\cdots,x_n) \in X^n$
    
    $$
    \sum \sum a_ia_jh(x_i,x_j) \geq0
    $$
    
    이는 일반적인 선형대수에서 정의하는 matrix의 positiveness하고 같은데, Matrix 또한 vector space 상에서는 Linear operator이므로, 결국 function의  postive definteness와 같은 의미라고 생각하면된다. 
    

**Positive definiteness in Hilbert Space**

Let F be any  Hilbert Space, and $\phi: X \rightarrow F$. Then $h(x,y):=<\phi(x),\phi(y)>_F$ is a positive definite functions.

Pf)

$$
\begin{align}
\sum \sum a_i a_jh(x_i,x_j)=\sum \sum &<a_i\phi(x_i),a_j\phi(x_j)>_F\\ 
&= <\sum a_i\phi(x_i), \sum a_j \phi(x_j)>_F\\
&= {\lVert \sum a_i \phi(x_i)\rVert^2}_F \geq 0
\end{align}
$$

thus, any kernel in hilbert space is positive definite

여기까지 Reproducing Kernel Hilbert Space의 개념적 정리에 대해 다뤄보았고, 

다음포스트에서는 Moore Arosjan Theorem부터, RKHS의 활용에 대해서 다뤄보도록 하겠습니다.