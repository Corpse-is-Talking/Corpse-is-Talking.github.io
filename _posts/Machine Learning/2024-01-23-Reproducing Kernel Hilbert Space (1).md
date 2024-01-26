---
title: "[ML]9.Reproducing kernel Hilbert Space (1)"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: RKHS(1) From Definition of Space to Riesz Representation Theorem.
use_math: true
comments: true
---
# Reproducing Kernel Hilbert Space (1)

## 1. 개요

석사 1기 진행중에 Reproducing kernel Hilbert space 라는 것을 배우게 되었는데, 스스로 정리도 해볼겸 작성해보게 되었습니다.

잘 정리된 영문 자료를 참고하였고,   

이번 포스트에서는 Normed Space, Inner Product Space, Cauchy Sequence, Completeness, Banach and Hilbert Space 를 정의하고,
RKHS의 기반이 되는 Riesz Representation Theorem 까지 다뤄보겠습니다.

다음 포스트부터는 RKHS의 정의 , 그 활용사례까지 차근차근정리해보도록 하겠습니다. 

한국말로 옮기거나 타이핑을 하면서 오타나 이상한점이 많을 수도 있습니다. 

## 2. 공간의 정의

RKHS(Reproducing Kernel Hilbert Space) 는 그 이름에도 들어있듯이, Hilbert Space 입니다. 여기서 Space는 vector space를 의미하고(or linear space) 벡터 공간에 대한 기본적인 이해는 선형대수에 잘정의 되있기 때문에 이를 참고하시면 좋을 것 같습니다. 

Norm~ Riesz representation Theorem

### **2.1 Norm**

Let $F$ be a vector Space over $\mathbb{R}$. A function $ {\lVert \cdot \rVert}_F : F-> [0,\infty]$ is said to be norm on $F$  if 

1.  ${\lVert f \rVert}_F =0$ if and only if $f=0$ (R 에서 정의된 vector space 에서 norm 이 0이면,  value 는 0이어야 한다.)
2. ${\lVert \lambda f \rVert}_F={\vert \lambda \rvert} {\lVert f \rVert}_F$  (positive homogenity)
3. ${\lVert f+g \rVert}_F \leq {\lVert f \rVert}_F+ {\lVert g \rVert}_F$ (triangular inequality) 

즉,  $F$ 라는 vector space에서 $ {\lVert \cdot \rVert}_F : F-> [0,\infty]$ 라는 function이 위 세가지를 만족하면서 정의 되면, function  $F$는 norm 이된다. 

### **2.2 Normed Vector Space**

Norm 이 정의된 vectorspace 를 Normed Vector space 라고 한다. 

### **2.3 Convergence and cauchy**

Vector Space에서의 Convergence 와 Cauchy 수열을 새롭게 정의한다. 

- **Convergence**
    
    A sequence of $\{ f_n\}_{n=1}^{\infty}$ of elements of a normed vector space  $F$ is said to converge to $f \in F$ if for every $\epsilon>0,$  there exists $N=N(\epsilon) \in \mathbb{N},$ such that for all $n \geq N, {\lVert f_n-f \rVert}_F<\epsilon$
    
    (Vector space에서의 수렴을 정의한다는 것을 제외하면 수렴의 정의와 비슷하다)
    

- **Cauchy sequence**
    
    A sequence of $\{ f_n\}_{n=1}^{\infty}$ of elements of a normed vector space  $F$ is said to be Cauchy if  for every $\epsilon>0,$  there exists $N=N(\epsilon) \in  \mathbb{N},$ such that for all $n,m \geq N, {\lVert f_n-f_m \rVert}_F<\epsilon$
    
    역시 Vectorspace에서의 Cauch 수열의 정의를 한 것이다. 
    
- **Note**
    
    Convergence Sequence is Cauchy but not every Cauchy Sequence  in every normed space converges. 
    
    Example: $1,1.4, 1.41, 1.414,\cdots$ 로 정의된 Cauchy 수열이 있다고 하자, 이 Cauchy 수열은 $\ Q$(유리수 집합) 에서는 수렴하지 않는다. ($\sqrt2$  는 유리수가 아니기 때문에)
    

### **2.4 Complete Space and Banach Space**

- **Complete Space**
    
    A space $X$ is complete if;f every Cauchyu sequence in $X$ converges, it has a limit, and this limit is in $X$.여기서 $X$ 는 당연히 vector space를 의미한다. 
    
    (모든 Cauchy Sequence가 Convergent한 Vectorspace→ Complete Space) 
    
- **Banach Space**
    
    Banach Space is complete normed space, that is, it contains the limits of all its Cauchy sequences. 
    
    (Complete Space Equipped with norm)
    
- **Complete Space를 정의하는 이유**
    
    Complete space(Cauchy Sequence가 수렴) 를 정의함으로써,  다음과 같은 표현을 할 수 있게 된다. 
    
    $$x(t)=\sum_{n=1}^{\infty}\alpha_n\psi_n(t)$$
    
    즉 ,  space 에있는 특정 vector(function)을 무한개의 vector로 표현이 가능하다는 것인데, 이는 나중 포스트에서 다시 언급하도록 하겠습니다. 
    

### **2.5 Inner Product Space**

- **Inner product Space**
    
    Let $F$ be a vector Space over $\mathbb{R}$. A function $<\cdot,\cdot>_F : F\times F-> \mathbb{R}$ is said to be inner product on $F$  if 
    
    1. $<\alpha_1f_1+ \alpha_2f_2,g>_F=\alpha_1<f_1,g>_F+\alpha_2<f_2,g>_F$
    2. $<f,g>_F=<g,f>_F$
    3. $<f,f>_F\geq0$ and $<f,f>_F=0$  if and only if $f=0$
    
    위 3가지 속성을 만족하는 function을 vector space $F$ 에서의 inner product라고 하고, innerproduct 가 정의된 Vector Space 를 inner Product space라고 한다. 
    
- **추가로**
    
    inner product induce norm but not every norm induce inner product. (EX L1norm)
    
    inner product는 norm도 유도할 수 있지만, 그역은 항상 성립하지 않는다. 
    

### **2.6 Hilbert Space**

- **Hilbert Space**
    
    Complete Inner Product Space (that is, Banach Space with Inner product ) 
    
- **Example:**
    
    For an index set $A,$ the space $l^2(A)$ is a Hilbert Space with inner product 
    
    $$
    <\{x_{\alpha}\},\{ y_{\alpha}\}>_{l^2{(A)}}=\sum x_{\alpha}y_{\alpha}
    $$
    
    inner product과 위와 같이 간단한 (우리가 일반적으로 아는 ) 곱의 형태로 정의되어있다면, $l^2(A)$ 는 Hilbert Space이다. 
    

### **2.7 Boundedness and Continuous linear operators**.

Hilbert Space의 정의를 위해서는 Boundedness와 Linear Operators(functions) 에 대한 이해가 필요하다. 두 개념의 이해를 위해 $F$ 와 $G$ 라는 normed vectorspace (vector space with norm defined) r공간을 정의하고 시작한다.

- **Linear Operators**
    
    A function $A:F \rightarrow G$ , where $F$ and $G$ are both normed linear spaces over $\mathbb{R}$ , is called  linear operator iff 
    
    1. Homogenity : $A(\alpha f)=\alpha(Af)$ for every $\alpha, f \in F$
    2. Additivity: $A(f+g)=Af+Ag \ f,g\in F$ 
    
    즉, $F$ 내에 존재하는 어떤 원소 $f$ 에대해서 Homogenity 와 Additivity를 만족하면서 변환을시켜쥬는 함수를 Linear Operator이라고 한다. 
    
- **Continuity**
    
    A function $A:F\rightarrow G$ is said to be continuous at $f_0 \in F$ if for every $\epsilon>0$  there exist $\delta=\delta(\epsilon,f_0)>0$  s.t 
    
    $$
    ||f-f_0||_F<\delta \ \ \ implies \ \ \ ||Af-Af_0||_G<\epsilon
    $$
    
    해석학에서 정의하는 연속이랑 정의가 비슷하고, $F$에서 정의된 norm과 $G$에서 정의된 norm이 달라도 성립한다. (벡터공간에서의 연속을 위에 Linear Operator을 도입하여 새롭게 정의한것이다. ) 다른말로 정의하면, $F$에서의 convergent 한 sequence가  $G$에서의 convergent sequence로 mapping된다고 생각하면 된다. 
    
- **Lipschitz continuity**
    
    A function $A:F\rightarrow G$ is said to be lipschitz continuous if for some $C$, s.t every $f_1,f_2 \in F, 
    ||Af_1-Af_2||_G \leq C||f_1-f_2||_F$
    
    It is clear that Lipschitz continuous → Uniform continuous($F$의 모든 instance에 대해서 연속)
    

- **Operator Norm and Boundedness**
    
    We define Operator Norm as, 
    
    $$ \lVert A \rVert =sup_{f\in F}\frac{ {\lVert Af \rVert}_G}{ {\lVert f \rVert}_F }$$
    
    We also define Boundedness
    
    The linear Operator $A:F\rightarrow G$ is said to be a bounded operator if $\lvert A \rvert <\infty$ (위에서 정의한 linearoperator의 norm이 상한이 존재하면, Bounded operator이라고 한다. )
    

### **2.8 Topological dual and Riesz Representation**

- **Topological Dual**
    
    If $F$ is a normed space, then the space $F'$ of continuous linear functionals $A: F\rightarrow \mathbb{R}$ is called the topological dual space of $F$ 
    
    즉, $F \rightarrow \mathbb{R}$ 로 mapping 해주는 모든 linear fucntionals 들로 이루어지는 공간을 $F$의 topological dual 이라고 한다. 
    
    *여기서 functionals 와 operator의 정의가 햇갈릴 수 있는데**, fucntionals는 벡터→ 스칼라, operatore은 벡터공간→ 벡터공간을 mapping**한다. 
    
- **Riesz representation theorem**
    
    In a Hilbert Space, $F$ , all continuous linear functionals are of the form $<\cdot,g>_F,$  for some $g \in F$. 
    
    즉, Hilbert space에 존재하는 어떤 원소(벡터)에 대해서 모든 linear functionals(function)은 $F$  에 존재하는 어떤 원소 $g$ 와의 inner product로 표현될 수 있다. ( Hilbert Space는 위에서 정의했듯이 innerproduct 라는 function이 define된 공간이므로 가능하다.)  
    
    Riesz representation의 핵심은 **모든 functional을 Innerproduct 의 형태로 나타낼 수 있다는 것이고 이것이 RKHS의 기반이 됩니다. 조금 애매할 수 있는데 functional→ inner product 로 나타낼 수 잇다는것에 집중을 하면 좋은 것 같습니다.** 
    
    Riesz Representation에 대한 증명은 너무 어려워서 생략하겠습니다 ㅎ…
    

### **2.9 Summary**

여기까지 정말 많은 definition들을 정의했는데, Hilbert Space가 어떻게 정의되는지, functionals 이나 linear operators 들에 대한 정의가 중요하고 햇갈릴 만한 포인트라고 생각합니다. 이제 이 정의들을 적제적소에 잘 활용하여 Reproducing Kernel hilbertr Space에 대해서 정의해보도록 하겠습니다. 

## 3. Reference

What is RKHS?

[https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/RKHS_Notes1.pdf](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/RKHS_Notes1.pdf)

각종 개념들에 대한 wikipidea.