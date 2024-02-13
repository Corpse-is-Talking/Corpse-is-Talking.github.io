---
title: "[ML]9.Reproducing kernel Hilbert Space (3)"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: RKHS(3), From Moore Arosjan Theorem to Examples of RKHS.
use_math: true
comments: true
---


이번 포스트에서는 지난 포스트까지 정의한 원론적인 RKHS에 대해 더 자세히 들어가려고 합니다. 

아마 위에 정의만으로는 그래서 도대체 이 RKHS가 뭔지 왜쓰는지 도대체 kernel은 왜 등장했는지 의문점을 가질 수 밖에 없다고 생각합니다.  그래서 이번 포스트에서는 RKHS(Reproducing Kernel Hilbert Space)를 구성하는 과정과  예시를 통해서 RKHS를 심층적으로 파보려고 합니다. 

## 개요

 그래서 이번포스트는, Moore Aroszjan Thorem과  RKHS의 몇가지 예시를 통해 그 실용성을 확인해보려고 합니다. 

## Making Reproducing Kernel Hilbert Space

**먼저 결론부터 이야기하자면 , Moore Aroszjan Theorem 에 의해서 모든 positive definite kernel은 적절한 RKHS를 구성 할 수 있습니다.**

이를 증명하는 과정에서 저자는 다음과 같은 방법을 활용합니다. 

- 특정 조건 두 가지가 성립하는 Hilbert Space $H_0$를 선언합니다.(Not RKHS,  but pre-RKHS, shoudl satisfy Inner product, Convergence of Cauchy Sequence in Norm )
- 선언한 Hilbert Space $H_0$에서 , $\lim_{n\rightarrow \infin} f_n \rightarrow f$이 되도록 하는 임의의 Cauchy Sequence $f_n$을 만들고, 그리고 이 $f$로 이루어진 공간 $H$를 새롭게 정의합니다.
- H가 RKHS임을 증명합니다.

이렇게 하는 이유는 다음과 같습니다. 

- positive Definite kernel로 부터 적절한 innerproduct를 정의하면, 손쉽게 $H_0$를 구성할 수 있는데,
    
    이런 정의들을 통해서 $H_0$를 구성할 수 있다면, 약간의 조건만 추가하여 RKHS를 정의할 수 있기 때문입니다. 이는 정확히 Moore Aroszjan Theroem 의 증명의 내용입니다. 
    

이번 포스트에서는 $H_0$를 통해서 정의된 $H$가 RKHS임을 증명하는 과정은 생략하고(, 내용이 너무 길어 보기 힘들어질 것 같다고 생각했습니다. ),Positive Definite Kernel을 통해서 $H_0$를 구성할 수 있는 것 정도만 보여드리고($H_0$를 구성할 수 있으면, RKHS또한 구성이 가능하므로) 그 이후는 예시로 보여드리도록 하겠습니다. 

### Moore Aroszjan Theorem

pdf→ positive definite

Definition : 

Let $k:X \times X \rightarrow \mathbb{R}$ be postiive definite,  then there is a unique RKHS $H \subset \mathbb{R}^X$ with reproducing kernel $k$ .

즉  pdf kernel을 통해서 → Unique한 RKHS인 $H$를 구성 가능하다는 것이 Moore Aroszjan Theorem입니다. 

그럼 이제 실제로 pdf커널을 통해 RKHS를 구성하는 과정을 확인해봅시다. 

Moore Aroszjan Theorem을 통해서 RKHS를 구성하기 전에 Pre-RKHS $H_0$를 구성하기 위해서 필요한 두가지 정리를 가져와 봅시다. 

1) Evaluation Functionals $\delta_x$ are continuous on $H_0$

2) Any Cauchy sequence $f_n$ in $H_0$ which converges pointwise to 0 also converges in $H_0$-norm to 0. 

즉, evaulation functionals가 continous하고, cauchy sequence 가 수렴한다면, 그 수렴값이  $H_0$ -norm과 같다는 것을 증명할 수 있다면,  $H_0$ 를 구성할 수 있고, 이를 통해서 위에서 언급한 RKHS를 구성 할 수 있다는 것입니다. ($H_0$를통해서 $H$를 구성하고, $H$가 rkhs임을 증명하는 과정은 참고자료 Section 4를 참고하시면 됩니다. )

### Spanning Pre-RKHS with PDF kernel

그럼 이제 positive definite kernel로 부터 Pre-RKHS를 구성해 봅시다. 

$H_0=span[k(\cdot,x)_{x \in X}]$ 를 통해서 구성하고, inner product를 아래와 같이 구성해 봅시다. 

$$
<f,g>_{H_0}=\sum_{i=1}^{n}\sum _{j=1}^{m}\alpha_i \beta_j k(x_i,y_j)
$$

이제 우리는 3가지만 증명하면 RKHS를 구성할 수 있게 됩니다. 

**1)** 구성한 inner product가 적절한 inner product인가 that is , does $<f,f>_{H_0} imply \ f=0$?
(여기서 조금더 구체적으로 증명할 필요가 있지만, 이 포스트에서는 생략하겠습니다, 자세한 증명은 reference 참고)

**2)** Evaluation functionals 가 Continuous한 것

**3)** Any Cauchy sequence $f_n$ in $H_0$ which converges pointwise to 0 also converges in $H_0$-norm to 0.

**pf of 1)** 

1)을 증명하기 위해서는 다음과 같은 증명된 lemma를 사용합니다. 

*If $h$ is positive definite, then,  $\lvert h(x_1,x_2)\rvert^4 \leq\lvert h(x_1,x_2)\rvert^2 h(x_1,x_1) h(x_2,x_2)$* **(증명는 참고자료 35 참고)**

따라서 , $<f,f>_{H_0}=0$  이라면, $f$ 도 0이 되어야 합니다. 

**pf of 2)**

이제 Evaluation Functionals $\delta_x$가 $H_0$에서 continuous 한 것을 증명해봅시다. 

Let $x \in X$ , $f=\sum \alpha_i k(\cdot,x_i)$

 

$$
<f,k(\cdot,x)>_{H_0}=\sum \alpha_i k(x,x_i)=f(x)
$$

and thus, for $f,g \in H_0$, 

$$
\lvert \delta_x(f)-\delta_x(g) \rvert=\lvert<f-g,k(\cdot,x)>_{H_0}\rvert \leq k^{0.5}(x,x) \lVert f-g \rVert_{H_0}
$$

Used Riesz Representation Theorem and this in turn tells that evaluation functionals $\delta_x$ is continous on $H_0$ (inequality는 Cauchy-Schwarz-Inequality 에 의해서, RKHS (1)번 포스트의 연속 내용 증명 참고. )

**pf of 3)**

take $\epsilon >0$ and define Cauchy { $f_n$ } in $H_0$ that converges pointwise to 0. Since Cauchy sequences are bounded, we may define $A>0$ s.t $\lVert f_n \rVert_{H_0}<A, \forall n \in N$. One can find $N_1 \in \mathbb{N}$ s.t $\lVert f_n-f_m\rVert_{H_0} <\epsilon/2A, for  \ n,m \geq N_1.$  Write $f_{N_1}=\sum \alpha k(\cdot,x_i)$. Take $N_2 \in \mathbb{N}$ s.t $n \geq N_2$ implies $\lvert f_n(x_n)\rvert < \frac{\epsilon}{2r\alpha_i}$ for all i , Now, for $n\geq max(N_1,N_2)$

$$
\begin{align}  \lVert f_n \rVert^2_{H_0}&\leq \lvert <f_n-f_{N_1},f_n>_{H_0}\rvert +\lvert<f_{N_1},f_n>_{H_0}\rvert \\ &\leq \lVert f_n-f_{N_1}\rVert_{H_0} \lVert f_n \rVert_{H_0}+\sum \lvert \alpha f_n(x_i)\rvert \\ &<\epsilon  \end{align}
$$

so $f_n$  converges to 0  in $\lVert \cdot \rVert_{H_0}.$  Thus, all the pre-RKHS axioms( inner product, evaluation functionals are continuous, cauchy sequence converge) are satisfied. 

증명은, $H_0$ 에서 0으로 수렴하는 코시수열(함수) $f_n$을 정의하고, Cauchy Sequence가 bounded 인 점을 이용하여, $f_n$이 $\lVert \cdot \rVert_{H_0}.$  에서 0으로 수렴하는 것을 증명합니다. (해석학을 배울때 사용하는 증명의 스킬들을 주로 사용하여 증명합니다. )

(여기서 햇갈릴 만한 포인트는 함수들의 pointwise convergence가 norm convergence가 다르다는 것입니다.  )  함수열, 함수열 의pointwise convergence에 대한 설명은 아래 블로그에 잘 설명이 되어 있습니다. 

[함수열 참고](https://blog.naver.com/at3650/220895665853)

이제 마지막으로 $H$의 reproducing kernel이 $k$
라는것을 확인해보면, 됩니다. Simply note that $f \in H$  and $f_n$ in $H_0$ converges to $f$ pointwise., 

$$
<f,k(\cdot,x)>_H=lim_{n\rightarrow \infin}f_n(x) =f(x)
$$

## Examples

RKHS는 function들로이루어진 Space이고 function들간의 inner product로 이루어진 space입니다. 저 개인적으로 이게 일반적으로 아는 vector가아닌 function의 관점으로 이해하는게 쉽지는 않았습니다. 따라서 몇가지 활용예시를 통해서 정리해보고자 합니다.

### 예시 1. Hilbert Space of Linear Functionals without bias

다음과 같은 bias가 없는 Hilbert Space $H$와 그에 해당하는 Inner product를 아래와 같이 정의해봅시다.

$H=\{f: f_\beta(x)=\beta_1x_1 +\beta_2 x_2,\ \beta \in \R \}$

$<f_\beta,f_\theta>=<\beta,\theta>=\beta_1\theta_1+\beta_2\theta_2$

다시 Reisze representation theorem에 의거하여 생각하면

$f_\beta(x)$ =$<f_\beta, f_\theta>$인 $f_\theta$가 Hilbert space안에 존재한다는 것입니다. 

예를 들어 $f_\beta(2,3)=$$2\beta_1+3\beta_2$ 인데, $f_\theta=2x_1+3x_2$로 정의하게 된다면, 

$f_\beta(2,3)=\beta_1*2+ \beta_2*3$ 

또 Kernel 관점에서 생각해보면, 해당space에서 이를 가능하게 하는 kernel은 유니크하게 존재하고, $k(x,z)=<x,z>$  로 쓰는 것도 가능합니다. 

### 예시 2. Hilbert Space of linear terms and cross-product.

다음과 같은 bias가 없는 Hilbert Space $H$와 그에 해당하는 Inner product를 아래와 같이 정의해봅시다. (예시 1과 거의 같습니다. )

$H=\{f: f_\beta(x)=\beta_1x_1 +\beta_2 x_2, \beta_3x_1x_2,  \ \beta \in \R \}$

$<f_\beta,f_\theta>=<\beta,\theta>=\beta_1\theta_1+\beta_2\theta_2$+$\beta_3\theta_3$

이 예제는  kernel의 관점에서 봐봅시다.  $f_\beta$ 와 $f_\beta(x)$ 는 엄연히 다른 것을 인지해야합니다. 

(  $f_\beta$는 함수 그 자체를 나타내지만,$f_\beta(x)$는 $f_\beta$의  $x$에서의 evaluation을 나타내기 때문입니다. )

먼저 feature mappping을 다음과 같이 정의해봅시다. 

$$
\phi(x)=[x_1 \ x_2 \ x_1x_2]^T
$$

그리고 kernel function을 앞선 포스트에서 했던 것 처럼 다음과 같이 정의해보죠

$$
k(x,y)=\phi(x)^t\phi(y)
$$

이전 포스트에서 kernel에서 정의했던 것 처럼 $k(\cdot, x)= \phi(x)$이고, Reproducing property(Riesz Representation Therorem) 에 의해서 

$$
f_\beta(x)=<f_\beta,\phi(x)>
$$

와 같이 표현이 가능해집니다. 

이문제에서 RKHS를 사용하는 것의 장점이 드러나는데, 두가지 변수 $x_1, x_2$를 사용하여, 상대적으로 고차원(위 문제에서는 interaction이 고려된 3차원)으로 mapping을 진행해줍니다. 이를 그림으로 표현하면 다음과 같습니다.

 <img src = "https://github.com/lookbackjh/pytorch_SEFS/assets/75593825/74f42e03-94d1-47f0-8530-6f3b9534e999.png" width="50%" height="50%">

위사진에서 보면 알 수 있듯이 RKHS를 사용하는것의 매우 큰 장점을 알 수 있는데, 2차원에서는 linear하게 seperable하지 않지만, 3차원에서는 linear하게 seperable한 형태로 바뀐다는 것입니다. 

즉 , RKHS를 적용하므로써, 적절한 featuremapping(kernel)이 정의 된다면, 조금 더 고차원에서의 분석이 가능해진다는 것입니다.

## Conclusion
이번 포스트에서는 Moore Arosjan Theorem으로부터 RKHS를 구성하는 과정과 RKHS의 예시들에 대해서 살펴보았습니다. 
대부분의 증명이나 내용은 첫번째 포스트에서 언급했던 [자료](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/RKHS_Notes1.pdf)에서 참고했습니다.
다음 포스트에서는 SVM 및 Kernel Regression에서 어떤식으로 RKHS를 활용하는지에 다루어보겠습니다. 
