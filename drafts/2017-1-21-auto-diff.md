---
layout: post
categories: [machine learning, deep learning, mathematics]
title: "A Hands-on Introduction to Automatic Differentiation"
image: /assets/images/ad-cover.png
twitter_text: "A Hands-on Introduction to Automatic Differentiation"
tags: [DeepLearning, MachineLearning, Math]
---

If you have any experience with machine learning or its recently trending sub-discipline of deep learning, you probably know that we usually use gradient descent and its variants in the process of training our models and neural networks. In the early days, and you might have caught a glimpse of that if you ever did [Andrew Ng's MOOC](https://www.coursera.org/learn/machine-learning) or went through [Michael Nielsen's book *Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/), we used to code the gradients of the models by hand. However, as the models got more complex this became more cumbersome and tedious! This is where the modern computation libraries like [Theano](http://deeplearning.net/software/theano/), [Torch](http://torch.ch/), [Tensorflow](https://www.tensorflow.org/), [CNTK](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/) and others shone! These libraries allow us to just write our math and they automatically provide us with the gradients we want, which allowed the rapid deployment of more and more complex architectures. These libraries are able to "automagically" obtain the gradient via a technique called *Automatic Differentiation*. This post is to demystify the technique of its "magic"!

This post is not a tutorial on how to build a production ready computational library, there so much detail and pitfalls that goes into that, some of which I'm sure I'm not even aware of! The main purpose of this post is to introduce the idea of automatic differentiation and to get a grounded feeling of how it works. For this reason and because it has a wider popularity among practitioners than other languages, we're going to be using python as our programming language instead of a more efficient languages for the task like C or C++. All the code we'll encounter through the post resides in this [GitHub repository](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff).

## A Tale of Two Methods

One of the most traditional ways to compute the derivative of a function, taught in almost all numerical computations courses, is the via the definition of the derivate itself:

$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h}$$

However, because we can't divide by zero without the computer rightfully yelling at us, we settle to having $h$ to be a very small value that is close to but not zero. We *truncate* the zero to an approximate value and this results in an error in the derivative's value that we call **the truncation error**. Dealing with real numbers inside a digital computer almost always suffers from an error and loss in precision due to the fact that the limited binary representation of the number in memory cannot accommodate for the exact real value of the number, so the number gets rounded off after some point to be stored in a binary format. We call such error **a roundoff error**.

Even with the roundoff error, the truncation error in the derivative calculation is not that big of a deal if our sole purpose is to just compute the derivative, but that's rarely the case! Usually we want to compute a derivative to use it in other operations, like in machine learning when the derivative is used to update the values of our model's weights, in this case the existence of an error becomes a concern. When the it starts to *propagate* through other operations that suffer their own truncation and roundoff errors, the original error starts to expand and results in the final operation being thrown off its exact value. This phenomenon is called **error propagation** and its one of the major problem that face such a way to compute derivatives. It's possible to reformulate the limit rule a bit to result in lass truncation error, but with every decrease in error we make we get an increase in the computational complexity needed to carry out the operation.

One other approach to compute the derivative of a function is by exploiting the fact that taking the derivative of a function is a purely mechanical process. Let's for example examine how we can take the derivative of the function:

$$f(x)=\left(\sin x\right)^{\sin x}$$

We can compute that derivative, a step-by-step, using the elementary rules of differentiation as follows:

$$
\begin{split}
    f'(x) & = \left(\left(\sin x\right)^{\sin x}\right)'  = \left(e^{\sin x \ln    \sin x}\right)' = \left(e^{u(x)}\right)' \\
    & = \frac{df}{du}\times\frac{du}{dx} = \frac{d}{du}\left(e^{u}\right) \times \frac{d}{dx}\left(\sin x \ln \sin x\right)& \text{(Chain Rule)} \\
    & = e^u \frac{d}{dx}\left(\sin x \ln \sin x\right) & \text{($e^x$ Rule)} \\
    & = e^u\left(\frac{d}{dx}\left(\ln \sin x\right) \sin x + \frac{d}{dx}\left(\sin x\right) \ln \sin x\right) & \text{(Product Rule)} \\
    & = e^u\left(\frac{d}{dx}\left(\ln v(x)\right) \sin x + \cos x \ln \sin x\right) & \text{($\sin$ Rule)} \\
    & = e^u\left(\left(\frac{d}{dv}\left(\ln v\right) \times \frac{d}{dx}(\sin x)\right) \sin x + \cos x \ln \sin x\right) & \text{(Chain Rule)} \\
    & = e^u\left(\left(\frac{1}{v}\frac{d}{dx}(\sin x)\right) \sin x + \cos x \ln \sin x\right) & \text{($\ln$ Rule)} \\
    & = e^u\left(\frac{\cos x}{v} \times \sin x + \cos x \ln \sin x\right) & \text{($\sin$ Rule)} \\
    & = \left(\sin x\right)^{\sin x}\left(\frac{\cos x}{\sin x} \times \sin x + \cos x \ln \sin x\right) & \text{(Substitution)}\\
    & \color{red} = \left(\sin x\right)^{\sin x}\left(\cos x + \cos x \ln \sin x\right) \\
    & \color{red} = \left(\sin x\right)^{\sin x}\cos x\left(1 + \ln \sin x\right)
\end{split}
$$

By observing the step-by-step solution, it appears how mechanical the differentiation process is and how it can be represented as an iterative application of a few simple rules. An approach to implement this is to construct some sort of an [abstract syntax tree (AST)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) of the given expressions, this tree is then traversed and each node gets transformed using the simple rules to another AST node representing the derivative of that node. The following figure shows this process for single multiplication node.
![](/assets/images/symb-diff-ast.png)

This method of computing the derivative is called **symbolic differentiation** as it manipulates the symbols of the expressions directly to arrive at the derivative. This method overcomes the error problem in our earlier method, because the final expression of the derivative will only suffer from the computer's roundoff error but there will be no truncation error; the value of the derivative will be as exact as it can be represented in a digital computer. As nice as this is, the symbolic differentiation suffers from two drawbacks.

Go back and take a look on the step-by-step solution of the derivative, notice the two red steps in the end? These steps are not immediately reachable by symbolic differentiation engine! Such engine would usually stop at the last black step without the simplification of the expression. It's obvious that evaluating the unsimplified expression is not as efficient as evaluating the simplified one; there are redundant (like the $\cos x$ in the two terms within the parentheses) and useless computations (like $\frac{1}{\sin x}\sin x$ in the first term between the parentheses) involved! This problem becomes more serious when the expression to differentiate becomes more complex and redundant and useless computations keep popping up all over the derivative. Implementing an optimizer that would simplify the derivative expression is not impossible, but it will introduce extra complexity in both the maintenance of such implementation and the computational cost.

Another drawback to the symbolic approach is that it doesn't lend itself nicely to programmatic constructs. Say that you created a function `transform(x)` that takes a vector of values and applies a series of calculations that involves control branches and loops to transform that vector into a single value. It's not really clear how to calculate the derivative of such transform using symbolic differentiation as it is clear with our earlier numerical approach. This kind of painful Trade-off between the two methods is one of the main motivators behind the adoption of the method that we'll start to investigative now, which is automatic differentiation (AD).

# Dual Numbers

Let's take a close look at the [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) of any function $f(x)$ around the point $x=a$:

$$f(x) = \sum_{k=0}^{\infty}\frac{f^{(k)}(a)}{k!}(x-a)^k = f(a) + f'(a)(x-a) + \sum_{k=2}^{\infty}\frac{f^{(k)}(a)}{k!}(x-a)^k$$

wouldn't it be nice if we could plug in some value of $x$ that would preserve us the term containing $f'(a)$ and make the rest of the infinite sum evaluate to exactly zero? In that way there would be not truncation error and we could get the value of the derivative by just evaluating the function itself! Unfortunately, it's easy to see that there's no real value that would achieve us that. The only two ways the infinite sum would evaluate to exactly zero would be either if all the higher-order derivatives $f^{(k)}(a)$ are zeros; in which case the method would not be general to all possible functions (it will actually be limited to only linear functions), or $(x-a)^k$ is zero which means that $x=a$; in which case the term including $f'(a)$ would also evaluate to zero and we lose the first derivative.

However, having no real solutions to a problem doesn't usually stop the mathematicians from finding a solution outside the real realm, and just as they constructed the set of complex numbers to account for the value of $\sqrt{-1}$, we can devise our own set of numbers to achieve our goal! We'll define this new set as the set of all numbers in the form $a + b\epsilon$ where $a,b \in \mathbb{R}$ and $\epsilon \neq 0$ while $\epsilon^2=0$ <span class='sidenote'>This cannot be satisfied with a single number like $\sqrt{-1}$ does for $i$, $\epsilon$ is a matrix whose square has all elements as zeros, something like $\epsilon = \binom{0 \hspace{1em} 1}{0 \hspace{1em} 0}$ for which $\epsilon^2 = \binom{0 \hspace{1em} 0}{0 \hspace{1em} 0}$. These matrices are called [nilpotent](https://en.wikipedia.org/wiki/Nilpotent_matrix) which means "without power".</span>. We'll call this new set the set of [Dual Numbers](https://en.wikipedia.org/wiki/Dual_number) and we'll denote $a$ as the real component and $b$ as the dual component. Now let's plug in $x = a + b\epsilon$ into our Taylor series and see what happens:

$$
\begin{split}
    f(a + b\epsilon) & = f(a) + f'(a)b\epsilon + \sum_{k=2}^{\infty}\frac{f^{(k)}(a)}{k!}b^k\epsilon^k \\
    & = f(a) + f'(a)b\epsilon + \underbrace{\epsilon^2\sum_{k=2}^{\infty}\frac{f^{(k)}(a)}{k!}b^k\epsilon^{k-2}}_{0} \\
    & = f(a) + f'(a)b\epsilon
\end{split}
$$

If we went ahead and set $b = 1$, we get that $f(a + \epsilon) = f(a) + f'(a)\epsilon$; which means that evaluating a function using dual number with a real component $a$ and dual component $1$ results in another dual number that has the value of $f(a)$ as its real component and "automatically" its derivative at $a$ as the dual component, no symbol manipulations, and no truncation errors! And if we generalized that to say that a dual number represents some value and its derivative, we can think of the representation of $x=a+b\epsilon$ as $a$ being the value of $x$ and $b$ as its derivative with respect to itself, which makes $b = 1$ as we set it earlier.

### Operations on Dual Numbers

Like any other number system, we can define a set of operations on its members. Like addition and multiplication for example:

$$(a + b\epsilon) + (c + d\epsilon) = (a + b) + (c + d)\epsilon$$

$$(a + b\epsilon)(c + d\epsilon) = ac + ad\epsilon + bc\epsilon + bd\epsilon^2 = ac + (bc + ad)\epsilon$$

The cool thing about those operations is that if we used them with the fact that for any function $f(x + \epsilon) = f(x) + f'(x)\epsilon$ we'll find that these operation "automatically" provide us with the simple rules of differentiation we're already familiar with! E.g. with the two rules of addition and multiplication we just derived we can get:

$$
\begin{split}
    f(x + \epsilon) + g(x + \epsilon) & = \left[f(x) + f'(x)\epsilon\right] + \left[g(x) + g'(x)\epsilon\right] \\
    & = \left[f(x) + g(x)\right] + \left[f'(x) + g'(x)\right]\epsilon
\end{split}
$$

$$
\begin{split}
    f(x + \epsilon)g(x + \epsilon) & = \left[f(x) + f'(x)\epsilon\right]\left[g(x) + g'(x)\epsilon\right] \\
    & = \left[f(x)g(x)\right] + \left[f'(x)g(x) + f(x)g'(x)\right]\epsilon
\end{split}
$$

Which reflect the addition and product rules of differentiation respectively! We could also derive more complex operations on dual numbers using the same fact that $f(a + b\epsilon) = f(a) + f'(a)b\epsilon$, for example:

$$\sin(a + b\epsilon) = \sin a + (\sin a)'b\epsilon = \sin a + (\cos a)b\epsilon$$

$$\ln(a + b\epsilon) = \ln a + (\ln a)'b\epsilon = \ln a + \frac{b}{a}\epsilon$$

The fact that we can derive such operations and that these operations "automatically" contain the necessary rules for differentiation makes the system of dual numbers the ideal candidate to build the first mode of automatic differentiation, which is called the *Forward Mode*.

# AD: Forward Mode

The first thing we need to make in order to implement the forward mode of AD is start by implementing the dual numbers themselves. We can simply implement them via a class that contains to real attributes, one for the real component and the other for the dual component.

```python
class DualNumber:

    def __init__(self, real, dual):

        self.real = real
        self.dual = dual
```

The important thing to do is to allow objects of this class to behave correctly with arithmetic operations in the way we just specified in the previous section. Such thing can be achieved with something known in programming languages as **Operator Overloading**. With operator overloading, a programming language allows us to change the functionality of pre-defined operators (such as +, -, \*, \*\* in python) according the type of arguments the operation takes, so say a + operator works in the ordinary way with ordinary real numbers while working in a different way with a user-defined type like that `DualNumber` class we just created.

In python, arithmetic operator overloading is done for some arbitrary class via implementing some [special named methods](https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types) reserved for numeric type emulation. For example, to overload the + operator, we need to implement a method called `__add__(self, other)` that takes our dual number object itself as an operand and another object for the other operand.

```python
from numbers import Number

class DualNumber:

    def __init__(self, real, dual): ...

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, Number):
            return DualNumber(self.real + other, self.dual)
        else:
            raise TypeError("Unsupported Type for __add__")
```
The way this works is very simple: we first check if the other operand is another dual number, if it is we return a new one with the sum of the real components and the sum of the dual components. If the other operator was just a real number, we return a new object with the same dual component and the real component with the new real operand added to it! If it's neither we just throw an error saying that this operand type is not supported. That way we could perform addition operations between dual numbers and themselves, and between dual numbers and real ones, at least when the real numbers comes on the right!

```python
x = DualNumber(4, 7)
y = DualNumber(-1, 5)

z = x + y  # valid operation: returns a DualNumber(3, 12)
w = x + 10 # valid operation: returns a DualNumber(14, 7)

v = 4 + y  # throws a TypeError
```

What's the problem was the last operation then? why does it break when the real number comes on the left of the operator?!

This has to do with the way python calls for the operator implementation. By default, the implementation of the operand to the left is the one called for the operator; and the left operand here has no idea how to deal with our `DualNumber` class! Python however provides a way around this: if the operand on the left has no idea how to operate on with the one on the right, the right one's implementation of a another special function called `__radd__(self,other)` is called instead (r for reverse). So in order to allow for addition operations where the real operand can come on the left, we need to implement that method into our class.

```python
from numbers import Number

class DualNumber:

    def __init__(self, real, dual): ...

    def __add__(self, other): ...

    def __radd__(self, other):
        # this time we know that other is a real and cannot be DualNumber
        return DualNumber(self.real + other, self.dual)
```  
You can see that this case is already included in the code we wrote for the regular `__add__` method, so in accordance to the DRY principle (Don't Repeat Yourself), we create a separate function that is capable of carrying out both modes of operation and make each of the special functions call that. This is particularly useful when the code for each of the modes tend to be big and using that abstraction cuts the amount of code in half (like in the overloading of the division operation). So our implementation for the addition operation would look like that:

```python
from numbers import Number

class DualNumber:

    def __init__(self, real, dual): ...

    def _add(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, Number):
            return DualNumber(self.real + other, self.dual)
        else:
            raise TypeError("Unsupported Type for __add__")

    def __add__(self, other):
        return self._add(other)

    def __radd__(self, other):
        return self._add(other)
```

However, a little caution needs to be taken when the operator is not commutative, like subtraction, division and power. In such case we need an additional flag that tells us if the dual number is coming first or not so that we could set that flag off in the reverse operation. Here's how the subtraction operation would look with such a flag.

```python
from numbers import Number

class DualNumber:

    ...

    def _sub(self, other, self_first=True):
        if self_first and isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        elif self_first and isinstance(other, Number):
            return DualNumber(self.real - other, self.dual)
        elif not self_first and isinstance(other, Number):
            return DualNumber(other - self.real, -1 * self.dual)
        else:
            raise TypeError("Unsupported Type for __sub__")

    def __sub__(self, other):
        return self._sub(other)

    def __rsub__(self, other):
        return self._sub(other, self_first=False)
```

The rest of the class implementation which also includes multiplication, division and powers, can be found in the code repository in the [dualnumbers.py](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/tree/master/dualnumbers/dualnumbers.py) file. It's fully commented and it follows exactly the same pattern of coding we used in the two operations here, the difference is in the math of each operation. The full derivation of these different math of can be found in [Dual-Algebra.pdf](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/tree/master/Dual-Algebra.pdf) file, also in the repository.

The next step after we defined what a dual number is and how it behaves is to create some mathematical functions that can handle the dual numbers. These can be easily defined as directly implementing similar math like those we got earlier for the $\sin$ and $\ln$ functions using the python's real math library `math`. An implementation for the $\sin$ function would go like this:

```python
from dualnumbers import DualNumber
import math as rmath

def sin(x):
    if isinstance(x, DualNumber):
        return DualNumber(rmath.sin(x.real), rmath.cos(x.real) * x.dual)
    else:
        return rmath.sin(x)
```

It simply checks if the argument is dual, if it is then it returns the dual evaluation of the function. Otherwise it just falls back on the real function. This would allow us to use these functions interchangeably between dual and real numbers. [dmath.py](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/tree/master/dualnumbers/dmath.py) file contains this method alongside a few other methods for different mathematical functions, the follow the same pattern as the `sin` method we just implemented and their specific math can also be found in the [Dual-Algebra.pdf](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/tree/master/Dual-Algebra.pdf) file. Both dmath.py and dualnumbers.py are contained in a package called `dualnumbers` and the repository contains a [Dual-Math notebook](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/tree/master/Dual-Math.ipynb) that showcases the different operations implemented in that package.

We're now ready to implement our forward mode of AD: all we need to do is take a function and the value of its variable at which we want to calculate the derivative, we then wrap this value into a dual number with a dual component of 1 and pass it to the function. The dual component of the resulting number is the derivative we want!

```python
from dualnumbers import DualNumber

def derivative(fx, x):
    return fx(DualNumber(x, 1)).dual
```

We implement this method into a module called `forward` within a package we'll call `autodiff`. Let's see how this works with the function we differentiated earlier: $f(x) = (\sin x)^{\sin x}$. At $x = \frac{\pi}{4}$ we can calculate the derivative from the resulting expression and find that it's $0.3616192241$, we'll also find that our `derivative` function returns the same value:

```python
from autodiff.forward import derivative
from math import pi
import dualnumbers.dmath as dmath


fx = lambda x: dmath.sin(x) ** dmath.sin(x)

print "%.10f" % (derivative(fx, pi/4))  # prints 0.3616192241
```

However, we're not going to have the derivative's expression every time to check it's value against the our `derivative` method, actually the point of having an AD system is to elevate the need to do derivations by hand! Instead, to ensure the correctness of our derivatives, we resort to a technique known in the Machine Learning field as **gradient checking**. In this technique we essentially compare the derivate from AD to a numerical derivate calculated using the limit method we saw first thing here. The limit method gives us an approximate value that we're sure of its correctness (because it is the definition of the derivative), so the value of our AD method should be close to the numerical one within an acceptable error defined by both [absolute and relative errors](https://en.wikipedia.org/wiki/Approximation_error).

```python
def check_derivative(fx, x, suspect):
    h = 1.e-7
    rerr = 1.e-5
    aerr = 1.e-8

    numerical_derivative = (fx(x + h) - fx(x)) / h
    accept_error = aerr + abs(numerical_derivative) * rerr

    return abs(suspect - numerical_derivative) <= accept_error
```

We can test that with a new function like $f(x) = x^22^x$ that we don't know its derivative at $x=\frac{1}{2}$ beforehand:

```python
from autodiff.forward import derivative, check_derivative

fx = lambda x: (x ** 2) * (2 ** x)

ad_derivative = derivative(fx, 0.5)

print "%.10f" % (ad_derivative)  # prints 1.6592780982
print "%s" % (check_derivative(fx, 0.5, ad_derivative))  # prints True
```

How about functions with multiple variables then? How can we compute the partial derivative with respect to each variable?

Remember that we treated the dual component of the variable as the variable's derivative with respect to the differentiation variable, which in the single variable case was itself and that made its dual component be set to 1. We can extended the same rational to include multiple variables; to get the partial derivative of a function with respect to a certain variable, we set that variable's dual component to 1 and all the other variables should have a dual component of 0 that represents their derivatives with respect to that variable. We can encourage that method by looking at the math of the simplest multi-variables function, the two variables function <span class='sidenote'>The math for a more general argument can be a little tedious to work with, that's why we suffice with the two variables case. However, a general argument could be made using [Taylor's series generalization in multiple variables](https://en.wikipedia.org/wiki/Taylor_series#Taylor_series_in_several_variables).</span>. For a function $f(x,y)$ evaluated with dual numbers $a+b\epsilon, c+d\epsilon$ for $x,y$ respectively, its Taylor series has the form:

$$f(a+b\epsilon, c+d\epsilon)= f(a, c) + \left(\frac{\partial f(a, c)}{\partial x}b + \frac{\partial f(a, c)}{\partial y}d\right)\epsilon$$

If we'd like to get the derivative with respect to $x$ we just set its dual component to 1 and $y$'s dual component to 0 as we just discussed, that would get us a dual number with just the partial derivative with respect to $x$ as its dual component:

$$f(a+\epsilon, c+0.\epsilon)= f(a, c) + \frac{\partial f(a, c)}{\partial x}\epsilon$$

The other way around would get us the partial derivative with respect to $y$. With that we can implement a more general version of our `derivative` method that could work with any function of any number of variables, we just specify the which variable we want to differentiate with respect to and we provide the list of variables' values instead of a single value.

```python
def derivative(fx, wrt, args):

    dual_args = []
    for i, arg in enumerate(args):
        if i == wrt:
            dual_args.append(DualNumber(arg, 1))
        else:
            dual_args.append(DualNumber(arg, 0))

    return fx(*dual_args).dual
```

Here `wrt` is the zero-based index of the variable we'd like to differentiate with respect to as it appears in arguments of the function `fx`, and `args` is a list of values representing the point we want to know the derivative at.

We can also generalize our `check_derivative` in the same way by adding $h$ only to the value of the variable we're differentiate with respect to.

```python
def check_derivative(fx, wrt, args, suspect):
    h = 1.e-7
    rerr = 1.e-5
    aerr = 1.e-8

    shifted_args = args[:]
    shifted_args[wrt] += h

    numerical_derivative = (fx(*shifted_args) - fx(*args)) / h
    accept_error = aerr + abs(numerical_derivative) * rerr

    return abs(suspect - numerical_derivative) <= accept_error
```

For example, let's see how that would work with a function like:

$$f(x,y,z) = \sin(x^{y + z}) - 3z\ln x^2y^3$$

we'd like the partial derivative with respect to $y$ at $(0.5, 4, -2.3)$:

```python
from autodiff.forward import derivative, check_derivative
import dualnumbers.dmath as dmath


f = lambda x,y,z: dmath.sin(x ** (y + z)) - 3 * z * dmath.log((x ** 2) * (y ** 3))
ad_derivative = derivative(f, 1, [0.5, 4, -2.3])

print "%.10f" % (ad_derivative)  # prints 4.9716845517
print "%s" % (check_derivative(f, 1, [0.5, 4, -2.3], ad_derivative))  # prints True
```

Back now to the original purpose we started this post with, which is computing gradients. The gradient of a function $f(\mathbf{x})$ where $\mathbf{x}$ is a vector of $n$ variables is the vector:

$$\nabla f = \begin{bmatrix}\frac{\partial f}{\partial x_1} & \dots & \frac{\partial f}{\partial x_n}\end{bmatrix}$$

We can easily compute that by calling our `derivative` function $n$ times on the function, each time with a different value for the `wrt` argument.

```python
def gradient(fx, args):

    grad = []
    for i,_ in enumerate(args):
        grad.append(derivative(fx, i, args))

    return grad
```

We could try this `gradient` method out using a [simple linear regression model](https://en.wikipedia.org/wiki/Simple_linear_regression) that we'll train using gradient descent. We'll be using a synthetic dataset of 500 points generated randomly from a noisy target $y = 1.4x - 0.7$. We'll use the mean-squared error as our loss measure in estimating the slope $\theta_1$ and the intercept $\theta_0$:

$$J(\theta_0, \theta_1) = \frac{1}{500}\sum_{i=1}^{500}\left(y - (\theta_1 x + \theta_0)\right)^2$$

and with a learning rate of $\alpha$, our iterative gradient-based update rules would be:

$$\theta_i = \theta_i - \alpha\frac{\partial}{\partial \theta_i}J(\theta_0, \theta_1)$$

```python
import random
from autodiff.forward import gradient

target = lambda x: 1.4 * x - 0.7 # synthizing model

# generating a synthetic dataset of 500 datapoints
Xs = [random.uniform(0, 1) for _ in range(500)]
ys = [target(x) + random.uniform(-1, 1) for x in Xs]  # target + noise

def loss(slope, intercept, alpha):
    loss_value = 0
    for i in range(500):
        loss_value += (ys[i] - (slope * Xs[i] + intercept)) ** 2

    return 0.002 * loss_value

# initial values of training parameters
slope = 0.1
intercept = 0
learning_rate = 0.1

# training loop
for _ in range(10000):
    grad = gradient(loss, [slope, intercept, 1])

    # update the parameters using gradient info
    slope -= learning_rate * grad[0]
    intercept -= learning_rate * grad[1]

# print the final values of the parameters
print "Slope %.2f" % (slope)  # prints "Slope 1.44"
print "Intercept %.2f" % (intercept)  # prints "Intercept -0.73"
```

By running this script (which can be found with all the previous ones in the [Forward-AD notebook](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/tree/master/Forward-AD.ipynb) in the repository), we can see that the final estimations are pretty good! Moreover, we were able to see how our forward AD methods works seamlessly with a programmatic construct like the `loss` function that involves control flow operations like loops, in an exact way with no truncation errors or inefficient symbol manipulations. However, calculating gradients with forward AD has an inefficiency of its own!

For a function $f(\mathbf{x})$ where $\mathbf{x}$ is a vector of $n$ variables; if the function itself has a computational complexity $O(K)$, then the gradient would have a complexity of $O(nK)$ which is not that of a trouble when $n$ is small. However, for something like a neural network where there could be hundreds of millions of parameters, hence $n$ is in the order of hundreds of millions, we'll be in big trouble! We need another AD method that could handle such a large numbers of parameters more efficiently!

# Starting from the End

Let's consider a function like $f(x) = \sin(2\ln x)$. Assume that all the calculators in the world disappeared in bizarre accident and we need to evaluate this function by hand! To evaluate it correctly we need to go through the following order of evaluation:

$$
f(x) = \underbrace{\sin(\overbrace{2\times\underbrace{\ln x}_{1}}^{2} )}_{3}
$$

Another way to look at this evaluation order is by decomposing the expression to simpler sequential calculations; so we could say that $w_0=\ln x$, $w_1=2w_0$, and finally $f = \sin w_1$. Using this decomposition we want to calculate the derivative $\frac{\partial f}{\partial x}$ when $x=2$ (we'll use the partial symbol $\partial$ for all derivatives from now on). Let's first go forward from start to finish and evaluate the values of the intermediate steps up to the value of $f$:

$$
w_0 = \ln 2 \approx 0.693 \rightarrow w_1 = 2w_0\approx 1.386 \rightarrow f=\sin w_1 \approx 0.983
$$

In our decomposition, we have $f$ as a function of $w_1$, so we start by calculating $\frac{\partial f}{\partial w_1}$ which equals to $\cos w_1 \approx 0.183$. We then have $w_1$ as a function of $w_0$, which makes $f$ implicitly a function of $w_0$ as well. Using the chain rule we can write:

$$\frac{\partial f}{\partial w_0} = \frac{\partial f}{\partial w_1}\frac{\partial w_1}{\partial w_0}$$

We already know $\frac{\partial f}{\partial w_1}$ form the previous step, and we can easily get $\frac{\partial w_1}{\partial w_0}$ from the definition of $w_1$, which is just $2$, giving us $\frac{\partial f}{\partial w_0}\approx 2\times0.183 \approx 0.366$. Taking this derivative and going one last step back to $w_0$, which is a direct function of $x$, we can write by also using the chain rule:

$$
    \frac{\partial f}{\partial x} = \frac{\partial f}{\partial w_0}\frac{\partial w_0}{\partial x}  = 0.366 \times \frac{1}{x}  \approx 0.366 \times 0.5 \approx 0.183
$$

A better way to look at this process is visually. Instead of writing down the intermediate steps like that, we visualize the whole operation as a **computational graph**, a [direct graph](https://en.wikipedia.org/wiki/Directed_graph) where the nodes represent variables, constants or simple binary/urinary operations; and the edges represent the flow of the the values from each node to the other. Our function at question here can be represented by the following computational graph:

![comp-graph-1](/assets/images/intro-graph.png)

Throughout the rest of the post, all the computational graphs we'll see will follow the same color code: lightblue for variables, orange for constants, and red for operations. We can see that this computational graph corresponds to the decomposition we made earlier, with the 'ln' node representing $w_0$, 'mul' node for $w_1$, and 'sin' node for $f$. Using the tool of computational graph, we can more visually see the process of propagating the derivative backward and applying the chain rule in th following animation:

<video src='/assets/videos/intro-ad.mp4'></video>

With this step-by-step animation, we can see how by traversing the computational graph in a [breadth-first](https://en.wikipedia.org/wiki/Breadth-first_search) manner starting from the node representing our final function, we can propagate the derivatives backwards until we reach the desired derivative. At each step, the current operation node (the one highlighted in green) propagates $f$'s derivative with respect to itself (the number written on the edge) to one of its operands nodes (the one at the other end of the edge); using the chain rule, $f$'s derivative w.r.t. the current operand node is evaluated and will be used in the next steps. We do not need to carry out this operation when the operand node is a constant, that's why the chain operation doesn't show when we process the edge leading to the constant 2.

So, following the path down the computational graph till we reach our variable gives us the derivative with respect to that variable. However, the examples we saw had only one path leading to the variable $x$, how about a function like $f(x) = x^2 + 2^x$? Let's see how the computational graph for this function looks like:

![two-paths-cg](/assets/images/two-paths-cg.png)

In such function, we have the variable $x$ contributing to two computational paths, so it will receive two derivatives when we start propagating the derivatives backwards, which poses a question about how the final derivative with respect to $x$ would look like! Maybe we can add the derivatives from the two paths to get the final one? While this sounds as just an answer based on simple intuition, it is actually the correct one! The rigorous base for this answer is what's called the **multivariable chain rule** <span class='sidenote'>[Here](https://www.youtube.com/watch?v=NO3AqAaAE6o)'s a very nice introduction on the multivariable chain rule from KhanAcademy</span>. In it's simplest form, which is the form we're concerned with here, the rule says that for a function $f$ that is a function of two other functions of $x$, that is $f = f(g(x), h(x))$, the derivative of $f$ with respect to $x$ is:

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g}\frac{\partial g}{\partial x} + \frac{\partial f}{\partial h}\frac{\partial h}{\partial x}
$$

By applying this idea to the backward propagation of derivatives in the computational graph of $x^2 + 2^x$ at $x=4$, we can see how the value of $\frac{\partial f}{\partial x}$ gets accumulated on as new propagated derivatives arrive at it:

<video src='/assets/videos/two-paths.mp4'></video>

This justifies why we initially set the derivatives at the variables (and at each other node) to zero. It also explains why we traverse the graph breadth-first; we want to make sure that all the contributions to a node's $f$ derivative (in AD lingo, this derivative is called the **adjoint**) has arrived before taking its value and propagating it further back through the graph.

A very important question you might have by this point is: **why bother?** Whats the point of going at the derivative from the other way around and keeping up with all that hassle?! The answer to this question becomes clear when we look at the same process applied to a multivariable function; for example, $f(x,y,z) = \sin(x+y) + xy^{z}$ at $(x,y,z) = (1, 2, 3)$.

<video src='/assets/videos/multi-ad.mp4'></video>

See what happened here? We were able to get the derivative with respect to all three variables in just a single run through the graph! So, if we have a function of $n$ variables that takes $O(K)$ time to traverse it's computational graph, it will take $O(K)$ time to obtain all the $n$ derivatives, no matter how big $n$ is! This is why we bother with going at the derivative from the end and backwards! This method gives us a mechanical way to automatically obtain the derivatives of a multivariable function without having to suffer the performance hit that forward mode AD had. This approach to differentiation is what constitutes the second mode of AD that we'll see now, the *reverse mode*.

# AD: Reverse Mode

The first thing we need to create in order to implement the reverse mode of AD is to create a way that would allow us to build computational graphs that represent the computations we express. There are two design choices when we go about implementing computational graphs; the first is to initially build the graph then run the necessary computations by feeding values to that graph, the other is to build a computational graph representation along with carrying out the calculations. The first choice, which is commonly referred to as static computational graphs, is what frameworks like TensorFlow and Torch use. The advantage of this choice is the ability to optimize the graph before running the calculations; big computational graphs usually could benefit from some optimizations that would allow the computations to be run faster and allow a more efficient usage of resources. A simple example of that could be found in the function $f(x) = (\sin x)^{\sin x}$; an optimizer on a static graph can turn it form the one on the left to the one on the right, with one less computation to worry about.

![static-opt](/assets/images/static-opt.png)

However, statically built graph are kind of hard to debug, and doesn't not play nicely with regular programming constructs like conditional branches and looping. On the other hand, the second choice builds the graph dynamically along with carrying the computations. Many new framework; like [PyTorch](http://pytorch.org/), [MinPy](https://minpy.readthedocs.io/en/latest/), and [Chainer](http://chainer.org/);  adapted this choice as it allows for a workflow that is much easier to understand and to debug, but it loses the ability to optimize the computations. The choice between the two is a trade-off between efficiency and simplicity, and because the goal here is to provide a simple introduction to the topic, we'll go with dynamic graphs as our design choice.

We are going to be using [numpy](http://www.numpy.org/) as our base computational engine off which we'll build the computational graphing tool. The essential element of a graph is a node; we can represent each node as an object and the edges could be specified by attributes in the node object pointing to other nodes. Because we want our nodes to be indifferent from regular numerical values, we'll start be defining a base `Node` class that extends numpy's essential data structure, the `ndarray`. In that way, we can get our nodes to behave exactly the same as ndarrays while having the ability to add the necessary extra functionalities we need to create the graphs. We follow numpy's [official guide](https://docs.scipy.org/doc/numpy-1.10.0/user/basics.subclassing.html) on extending the `ndarray` object.

```python
class Node(np.ndarray):

    def __new__(subtype, shape,
                dtype=float,
                buffer=None,
                offset=0,
                strides=None,
                order=None):

        newobj = np.ndarray.__new__(
            subtype, shape, dtype,
            buffer, offset, strides,
            order
        )

        return newobj
```

from this object we extend three other classes that represent the three types of nodes we saw in the earlier graphs; an `OperationalNode`, a `ConstantNode` and a `VariableNode`. These extensions are fairly simple and only has one addition to the base `Node` class, which is a static method called `create_using`. This method allows us to create nodes on the fly using a numpy's `ndarray` or a number without needing to pass the arguments of the base's `__new__` method separately, we let this method take care of that and also add any necessary extra attributes to the object. We can first see this in action with the `VariableNode` class in which the `create_using` method takes a number or an `ndarray` value along with an optional name and returns an `VariableNode` object initialized at that value with a name attribute pointing to the name given or an auto-generated name if none is given.    

```python
class VariableNode(Node):

     # a static attribute to count the unnamed instances
     count = 0

     @staticmethod
     def create_using(val, name=None):

        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)

        obj = VariableNode(
            strides=val.strides,
            shape=val.shape,
            dtype=val.dtype,
            buffer=val
        )
        if name is not None:
            obj.name = name
        else:
            obj.name = "var_%d" % (VariableNode.count)
            VariableNode.count += 1

        return obj

```

The `ConstantNode` class looks exactly like the `VariableNode` class except for the fact that we use **"const_"** instead of **"var_"** in the auto generation of the node's name; we created separate classes for them just to be able to distinguish between them in runtime, but in practice the `ConstantNode` would need more additions, like for example, some overloads for the +=, -=, \*=, and /= operators to prevent the modification of the constant's initialized value, but we're dropping that here.

The last type of nodes is the `OperationalNode` class. We want an operational node to know which operation it reflects (addition, subtraction, multiplication, ... etc), what is the result value of that operation, what are the operands nodes, and to have a name just like the other nodes have. Because of these requirements,the `create_using` method of the operational node looks a bit different than in the others.

```python
class OperationalNode(Node):

    # a static attribute to count for unnamed nodes
    nodes_counter = {}

    @staticmethod
    def create_using(opresult, opname, operand_a, operand_b=None, name=None):

        obj = OperationalNode(
            strides=opresult.strides,
            shape=opresult.shape,
            dtype=opresult.dtype,
            buffer=opresult
        )

        obj.opname = opname
        obj.operand_a = operand_a
        obj.operand_b = operand_b

        if name is not None:
            obj.name = name
        else:
            if opname not in OperationalNode.nodes_counter:
                OperationalNode.nodes_counter[opname] = 0

            node_id = OperationalNode.nodes_counter[opname]
            OperationalNode.nodes_counter[opname] += 1
            obj.name = "%s_%d" % (opname, node_id)

        return obj
```

Here, instead of having just a single static counter, we have a static dictionary of counters with each item having a key of one of the possible `opname` (add, sub, mul, ... etc) and a value holding the count of such operations in the graph. The `operand_b` argument is made optional to allow for operations that take a single operand such as $\exp$, $\sin$, $\ln$, ... etc. The `opresult` argument takes the final value of the operation, so our operational node is a just a representation of the operation, its operands and and its result; it doesn't not carry any operation like you would expect in a static computational graph framework. It only serves as a data structure we could run the reverse mode AD on.

The next thing we need to do build our computational graph module is to make sure that carrying out operations that involve the `Node` object (or any of its subclasses) also creates the operational nodes that represent these operations. In order to do that, we need to overload the the basic arithmetic operators of the `Node` class (and subsequently, all its subclasses) in the same way we did with the dual numbers implementation, but this time we need the operations to return instances of `OperationalNode` that correspond to it. To be able to do that while allowing our classes to use the same computational engine used originally by numpy's ndarrays, we create a method called `_nodify` that takes the name of the overloaded operation, say for example **__add__**, calls the original numpy **__add__** method to get the value of the operation then returns an `OperationalNode` reflecting it.

```python
class Node(np.ndarray):

    def __new__(subtype, shape, ...): ...

    def _nodify(self, method_name, other, opname, self_first=True):

        if not isinstance(other, Node):
            other = ConstantNode.create_using(other)
        opvalue = getattr(np.ndarray, method_name)(self, other)

        return OperationalNode.create_using(opvalue, opname,
            self if self_first else other,
            other if self_first else self
        )
```

The method also takes care of the other operand and transform it into a constant node if it's an instance of the `Node` class, this is to make sure that everything is correctly and fully represented in the graph. The `self_first` serves a similar purpose as in the dual numbers implementation; to put the operands in the correct order for non commutative operations. Now, with this method, we're ready to overload the operators on the `Node` class easily.

```python
class Node(np.ndarray):

    def __new__(subtype, shape, ...): ...

    def _nodify(self, method_name, other, opname, self_first=True): ...

    def __add__(self, other):
        return self._nodify('__add__', other, 'add')

    def __radd__(self, other):
        return self._nodify('__radd__', other, 'add')

    def __sub__(self, other):
        return self._nodify('__sub__', other, 'sub')

    def __rsub__(self, other):
        return self._nodify('__rsub__', other, 'sub', False)

        ...
```
More operations (including the transpose operations `ndarray.T`) are overloaded in the exact same way in the full implementation in the [nodes.py](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/blob/master/compgraph/nodes.py) file in the repository.

The last thing we're left to do in order to complete our computational graph framework is to create more operations and primitives that support computational graphs and would allow us to easily define their nodes, much like we did in the [dmath.py](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/tree/master/dualnumbers/dmath.py) file when we worked with dual numbers. We start with two essential methods that would allow us to create `ConstantNode`s and `VariableNode`s on the fly using some numerical value, without having to directly invoke the `create_using` method and working with the classes themselves.

```python
def variable(initial_value, name=None):
    return VariableNode.create_using(initial_value, name)


def constant(value, name=None):
    return ConstantNode.create_using(value, name)
```

We're now left with creating some interesting operations to support in the computational graph framework. This is fairly simple to do; we just get the inputs, which are supposedly instances of the `Node` class or its subclasses (if not, we create an appropriate `ConstantNode` for the given value), run the desired operation using regular numpy methods, then create and return an `OperationalNode` that with that value and these inputs as operands. The following are examples of that way on the summation operation and the dot product operation.

```python
def sum(array, axis=None, keepdims=False, name=None):
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.sum(array, axis=axis, keepdims=keepdims)

    return OperationalNode.create_using(opvalue, 'sum', array, name=name)


def dot(array_a, array_b, name=None):
    if not isinstance(array_a, Node):
        array_a = ConstantNode.create_using(array_a)
    if not isinstance(array_b, Node):
        array_b = ConstantNode.create_using(array_b)
    opvalue = np.dot(array_a, array_b)

    return OperationalNode.create_using(opvalue, 'dot', array_a, array_b, name)
```
More operations are implemented in the exact same way in the [api.py](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/blob/master/compgraph/api.py). Both the api.py file and the nodes.py file are packaged in the `compgraph` package. An additional visualization module is provided within that package to help visualizing the computational graphs created via a method called `visualize_at` which simply takes a `Node` object and draws the whole graph leading to it. This visualization method, along with all the methods defined in the api.py module are directly accessible from the `compgraph` package. The following snippet demonstrates how it can be used.

```python
import compgraph as cg

x = cg.variable(0.5, 'x')
y = cg.variable(4, 'y')
z = cg.variable(-2.3, 'z')

f = cg.sin(x ** (y + z)) - 3 * cg.log((x ** 2) * (y ** 3))

print "f = %s" % (f)  #prints 'f = -8.01481664426'

cg.visualize_at(f)
```
The call to the `visualize_at` method in the end of the snippet generates the following image of the graph nodes starting from the variables up to the `f` node

![](/assets/images/vis_result.png)

This example, among others, can be seen in the [Computational Graphs Notebook](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/blob/master/Computational%20Graphs.ipynb) in the repository. It's very recommended that you experiment with these examples and even create your own and visualize them to get a better handle on what's going on.

Now that we have a framework that would allow us to build computational graphs as we go on carrying out our operations, all we need now is something to carry out the actual AD operation: that is something applying the chain rule in breadth-first manner starting form the result node back to the variables nodes. Implementing a breadth-first traversal is as simple as starting from the target node and adding its previous nodes in first-in-first-out queue and then applying the same operation on the front node in the queue until the it gets empty, i.e. it reaches variable or constant nodes which have no previous nodes.

But before we go on defining a method applying that breadth-first traversal, we need a way to define the gradients of the diverse set of operations we have in a consistent way that would allow the traversal method to easily get the desired gradients (or adjoint) once it identified the operation's name. We can do this by standardizing the way we define our gradients method for all the operations we have, hence providing a consistent interface for the traversal method to use without change across all possible operation nodes.

![](/assets/images/grads_op.png)

The figure above depicts a way to standardize the gradients method: for each operation we define a method with the name `opname_grad` where *opname* is the operations name as its defined in the `compgraph` package (like *add, div, sum, dot, ...* etc). This method should take two arguments and returns a list of two objects: it should take the node's adjoint and the node object itself, and it should return the adjoints of its operand nodes; if it's a unary operation taking only one operand, then the other adjoint should be `None`. For example, the multiplication operation could be simply defined as:

```python
def mul_grad(prev_adjoint, node):
    return [
        prev_adjoint * node.operand_b,
        prev_adjoint * node.operand_a
    ]
```

Most of the gradient methods are just a simple application of the chain rule along with the basic differentiation operations like we see in `mul_grad`. However, when it comes to dealing with multi-dimensional arrays operations provided by numpy's `ndarray`, things can get a little tricky! For our purposes here, we can distinguish between two types operations that deal with the `ndarray`s: *reduction* and *arithmetic* operations.

#### Reduction Operations

In reduction operations, we take an `ndarray` and reduce it another form, possibly a smaller form, of it. An obvious example of that operations is the `sum` operation, which takes the whole array and reduce it to a single value representing the summation of its elements. Another one is the `max` operation that reduces the array to only the maximum value among the elements. The key point in defining the gradients of such operations is realizing that only the elements that contribute to the value of the operation should have a non-zero derivative of the operation with respect to it; the value of these derivatives is then defined by the arithmetic of the reduction operation itself.

![](/assets/images/sum_grad.png)
![](/assets/images/max_grad.png)

```python
from collections import defaultdict
from compgraph.nodes import *
import grads

def gradient(node):

    adjoint = defaultdict(int)
    grad = {}
    queue = NodesQueue()

    # put the given node in the queue and set its adjoint to one
    adjoint[node.name] = ConstantNode.create_using(np.ones(node.shape))
    queue.push(node)

    while len(queue) > 0:
        current_node = queue.pop()

        if isinstance(current_node, ConstantNode):
            continue
        if isinstance(current_node, VariableNode):
            grad[current_node.name] = adjoint[current_node.name]
            continue

        current_adjoint = adjoint[current_node.name]
        current_op = current_node.opname

        op_grad = getattr(grads, '%s_grad' % (current_op))
        next_adjoints = op_grad(current_adjoint, current_node)

        adjoint[current_node.operand_a.name] = grads.unbroadcast_adjoint(
            current_node.operand_a,
            adjoint[current_node.operand_a.name] + next_adjoints[0]
        )
        if current_node.operand_a not in queue:
            queue.push(current_node.operand_a)

        if current_node.operand_b is not None:
            adjoint[current_node.operand_b.name] = grads.unbroadcast_adjoint(
                current_node.operand_b,
                adjoint[current_node.operand_b.name] + next_adjoints[1]
            )
            if current_node.operand_b not in queue:
                queue.push(current_node.operand_b)

    return grad
```

{% include side-notes.html %}
{% include minimal-vid.html %}
