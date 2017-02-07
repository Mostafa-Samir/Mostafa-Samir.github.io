---
layout: post
categories: [machine learning, deep learning, mathematics]
title: "A Hands-on Introduction to Automatic Differentiation"
image: /assets/images/bias-var.png
twitter_text: "A Hands-on Introduction to Automatic Differentiation"
tags: [DeepLearning, MachineLearning, Math]
---

If you have any experience with machine learning or its recently trending sub-discipline of deep learning, you probably know that we usually use gradient descent and its variants in the process of training our models and neural networks. In the early days, and you might have caught a glimpse of that if you ever did [Andrew Ng's MOOC](https://www.coursera.org/learn/machine-learning) or went through [Michael Nielsen's book *Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/), we used to code the gradients of the models by hand. However, as the models got more complex this became more cumbersome and tedious! This is where the modern computation libraries like [Theano](http://deeplearning.net/software/theano/), [Torch](http://torch.ch/), [Tensorflow](https://www.tensorflow.org/), [CNTK](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/) and others shone! These libraries allow us to just write our math and they automatically provide us with the gradients we want, which allowed the rapid deployment of more and more complex architectures. These libraries are able to "automagically" obtain the gradient via a technique called *Automatic Differentiation*. This post is to demystify the technique of its "magic"!

This post is not a tutorial on how to build a production ready computational library, there so much detail and pitfalls that goes into that, some of which I'm sure I'm not even aware of! The main purpose of this post is to introduce the idea of automatic differentiation and to get a grounded feeling of how it works. For this reason and because it has a wider popularity among practitioners than other languages, we're going to be using python as our programming language instead of a more efficient languages for the task like C or C++. All the code we'll encounter through the post resides in this [GitHub repo](<put-repo-link-here>).

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

The rest of the class implementation which also includes multiplication, division and powers, can be found in the code repository in the [dualnumbers.py](<link to file in repo>) file. It's fully commented and it follows exactly the same pattern of coding we used in the two operations here, the difference is in the math of each operation. The full derivation of these different math of can be found in [Dual Algebra.pdf](<link to file in the repo>) file, also in the repository.

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

It simply checks if the argument is dual, if it is then it returns the dual evaluation of the function. Otherwise it just falls back on the real function. This would allow us to use these functions interchangeably between dual and real numbers. [dmath.py](<link to file in the repo>) file contains this method alongside a few other methods for different mathematical functions, the follow the same pattern as the `sin` method we just implemented and their specific math can also be found in the [Dual Algebra.pdf](<link to file in the repo>) file. Both dmath.py and dualnumbers.py are contained in a package called `dualnumbers` and the repository contains a file [dual-math.py](<link to file in repo>) that showcases the different operations implemented in that package.


{% include side-notes.html %}
