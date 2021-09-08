---
title: "[ML]4. Gradient Descent"
categories:
  - MachineLearning
tags:
  - ML
  - python
  - Industrial Engineering
  - Blog
excerpt: Gradient Descent and Stochastic Gradient Descent
use_math: true
comments: true

---
## 0. Review
지난 포스트까지는 기본적인 Statistical Learning Theory의 Framework에 대해서 알아보았다. 이번 포스트부터는 이 프레임워크들을 이용해서,  ML에서 가장 많이 사용되는 방법 중하나인 Gradient Descent, 더나아가 Gradient descent의 효율을 높여주는 Backtracking Line search 까지 알아보겠다.

## 1. Gradient Descent

- #### Definition.
    __Gradient descent__ is a first-order iterative optimization algorithm for finding a __local minimum__ of a differentiable function. (Wikipidea)
    위키피디아의 정의 따르면, __Gradient descent__ 는 미분 가능한 함수의 1차미분계수를 통해, 그 함수의 __local minimum__ 을 찾는 알고리즘이다.


    미분가능한 다변수함수 $F(X)$에 대하여, 어떤 점 X에서,  __Gradient(1차 미분계수)__ $-\nabla F$는, __현재 위치 X 가 가장 빠르게 감소하는 방향을 나타내준다.__ 이를 이용하여, Step 사이즈를 작게하여, X를 조금씩 움직여서, local minimum 에 도달할때까지 반복하는 것이다. 




    알고리즘을 수학적으로 살펴 보면 다음과 같다.
    


  
    For differentiable multi-variable function $F(X)$, very small $\gamma>0$ , with starting point $x_0$, 

    Until Convergence to local minumum, Repeat


    $$x_{n+1}= x_n-\gamma \nabla F(x_n)$$ 
    

  <img src = "https://user-images.githubusercontent.com/75593825/130398636-bee6cddf-0d78-4c93-b645-9dc9b6556228.png" width="50%" height="50%">
  [Gradient Descent 예시, 출처: 위키피디아](https://en.wikipedia.org/wiki/Gradient_descent)


  그림과같이 Contour Line  (같은 F값을 가지는 X들의 집합 ,바깥쪽 원이 큰 값을 가짐)이 그려져 있고,초기점 $x_0$ 에서 시작하여 Iteration이 진행될수록 local minimum 에 도달해나가는 모습을 볼 수 있다.

## 2. Convergence Theroem for fixed step size..

- Definition:
  ![image](https://user-images.githubusercontent.com/75593825/132162353-7e9b4f3f-10a2-40f1-8a9e-8d92a6e4a502.png)
  만약 $f$ 가 Convex, Differentiable, Lipschitz continuous with continuous with constant L을 만족 할 때, step size t를 1/L 보다 작게 해서 gradient descent 를 수행한다면, 결과값은 항상 수렴한다는것이다. 여기서 Convex 와 Lipshcitz continuous 에 대해서 더 자세히 살펴보자

  1. Convex function

      Definition:    $\forall x_1,x_2\in X,\forall \alpha \in [0,1] $
      $$ f(\alpha x_1+(1−\alpha )x_2)≤\alpha f(x1)+(1−\alpha)f(x_2)) $$
      을 만족하면, $f$는 Convex function 이라고 한다. 기하학적으로는 그래프 상의 임의 의 두 점 x,y 에 대히서 x와 y를 잇는 선분을 그렸을 때,  이 선분은 __항상__ 그래프보다 크거나 같게 위치한다.


      ![image](https://user-images.githubusercontent.com/75593825/132164231-e4c316c7-a00a-4c0a-a112-eb4a6eb2088b.png) 
      [출처: Convex Optimization, Stephen Voyd, Lieven Vandenberghe]



      2차원에서 생각해보면, 아래로볼록한 함수, 흔히 언급되는 $f(x)=x^2$ 같은 함수가 Convex Function 이다. Convex fucntion에 대해서는 나중에 더 자세히 다루겠지만, 가장 중요한 부분만 짚고 넘어가겠다. 
      __Convex 함수의 Local minumum은  global minimum 이다.__, 만약 Strongly Convex 하다면(위의 Definition에서 부등호가 $\leq$ 가아니라 $<$ 이면 ) __Unique__ 하게 global minumum 이 존재한다.
      즉, 함수가 Convex 하다면, 함수의 최솟값이(유일하지는 않은) 존재 한다는 것이다. 

  2. Lipshcitz continuous 
      
      Gradient Descent 문제에서, Lipschctiz Continous 조건이 필요한 이유는 수렴성의 여부 때문인데, 만약 Step size를 너무 크게 잡아버린다면, 아무리 Gradient Descent 를 반복해도, 원하는 값에 도달하지 못할 수가 있다. 

      ![image](https://user-images.githubusercontent.com/75593825/132165076-bf7fda6c-f6d8-43ea-9bb0-e50ff45e4de4.png)

      (왼쪽, Stepsize를 작게잡는 경우, 도달은 하지만 너무느림, 오른쪽그림, Stepsize를 크게잡은 경우 원하는 값에 도달을 못함)[출처](http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf)
      
      위 Lipschitz Continous의 정의는  적당한 Fixed Step size를 정했을 때, gradient descent 가 원하는 값에 수렴 하는것을 보장해준다. 증명은 [여기](http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf)

  이제 우리는, 함수가 Convex 하다면, Global minimum 이 존재한다는 것을 알았고, 적당한 Stepsize를 잡으면, Optimal 에 수렴한다는 사실도 알았다. 
  하지만, 강의에서는 Practical 한 상황에서  $\frac{1}{L}$을 Step size 로 잡게 된다면, 수렴은 보장되지만, 위에 사진처럼 수렴이 너무 느린 경우가 많다고 한다. 따라서, 더 큰 Step size를 잡아도 된다고 한다.  

- #### Stop Condition

  Iterate until $\parallel \nabla f(x) \parallel_2 \ \leq \epsilon	$  for $\epsilon>0$ you choose. ( $\nabla f$=0 이라는 것은 local minimum 이라는 뜻이고, convex한 상황에서는 local min= global min 
  )	
  Step size를 너무 크게 잡아 수렴 하지 않을 것 같을 경우에는 멈추고, Step size 를 변경 혹은 Backtracking 등 다른 방법을 이용.

## 3.  Backtracking Line Search
  
  Gradient Descent는 알고리즘을 그대로 이용하면 매우 느리다고 한다, 따라서 알고리즘에 조금씩 변화를 주면서 성능을 크게 향상시키는 여러가지 방법들이 있는데, 그 중에서 Backtracking Line Search 에 대해서 알아보겠다. 

 -  Definition:

    Gradient Descent 의 Step 을 진행하면서, 만약 현재 점에서 다음 점으로 갈 때, 너무 많이 갔다고 판단되면, 되돌아오고, 아니면 그대로 진행해서 효율을 증가시켜주는 방법이다.


    ![image](https://user-images.githubusercontent.com/75593825/132169207-582ab76e-548c-4c9c-b51f-cfcf483b12df.png)
    [출처: Convex Optimization, Stephen Voyd, Lieven Vandenberghe]


    이 사진에서 유의 깊게 봐야할것은 $\alpha$ 를 곱한 점선함수와, $f(x+t\Delta x)$ 이다. $f(x+t\Delta x)$는 원래 점 x에서 이동했을때의 함수값을 가리키는데 이 함수값의 위치에 따라서, Backtracking의 과정이 변한다. 

    위 그림에서 t는 Step size로 Step size 에 따라서 변하는 f의 값을 그래프로 나타낸 것이다. 

    첫번째 접선은, 현재점 x에서 그린 접선인데, t를 어떻게 잡아도 항상 접선 위에 있으므로, t를 잘 잡았는지의 여부를 판단 할 수 없다. 

    두번째 접선은 접선의 기울기에 $\alpha$를 곱해서 구한 직선인데,  $f(x+t\Delta x)$가 이 점선 보다 위에있으면, 많이갔다고 판단해서 Stepsize를 줄여 점선아래로 오게 만들고, 점선 아래에 있으면, 적당히 잘 갔다고 판단한다.

    알고리즘:

    ![image](https://user-images.githubusercontent.com/75593825/132449982-3440f4e0-9c52-409d-9099-9eb0304ee226.png)
    
    만약, t=1로 초기화하고, 이동했을 때의 함수값이 $\alpha$를 곱한 접선의 함수값보다 크다면( 위 그림에서 점선의 위치보다 높다면), $t=\beta t$ 를해주어, 이동하는 값을 줄이는 것이다. 그후 조건이 충족된다면(위 그림에서 점선의 위치보다 낮아진다면), Gradient descent를 한 step 진행시킨다.


    ![image](https://user-images.githubusercontent.com/75593825/132451145-0970e873-ffa8-4058-bf83-6bff69279bd7.png)
    
    이후는 통상적인 Gradient Descent 와 똑같이 진행하면 된다. 원하는 t를 Backtracking Line Search 를 통해서 구하고, 위에서 언급했듯이 
    $\parallel \nabla f(x) \parallel_2 \ \leq \epsilon	$ 이 될때까지 반복하면 된다. 




- Backtracking Termination :

  $f$는 Convex 하므로, $\nabla f(x)^T\nabla x$ <0 따라서, [Linear Approximation](https://en.wikipedia.org/wiki/Linear_approximation)을 이용하면, $t$ 가 매우 작을 때


  $$ f(x+t\nabla x )\approx f(x) +t\nabla f(x)^T \nabla x < f(x)+\alpha t \nabla f(x)^T \nabla x$$
  
  를 만족하는데, 이말은 t에 계속해서 $\beta$를 곱해 감소시키면($0<\beta<1$), t는 0에 근접하므로, 결국에는  $f(x+t\nabla x )$ 가
  $ f(x)+\alpha t \nabla f(x)^T \nabla x $ 보다 작아진다는 것이다. 


## 4. 참고문헌

  Convex Optimization Chapter 9, Stephen Voyd, Lieven Vandenberghe

  [Gradient Descent, Lecture note from utexas](http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf)

  [Introduction to Statistical Learning Theory, Sgd lecture note](https://davidrosenberg.github.io/mlcourse/Archive/2017Fall/Lectures/02b.SGD.pdf)

  [모두를 위한 컨벡스 최적화](https://wikidocs.net/17052)





  



    
 