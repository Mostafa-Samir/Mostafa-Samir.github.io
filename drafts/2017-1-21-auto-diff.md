---
layout: post
categories: [machine learning, deep learning, mathematics]
title: "A Hands-on Introduction to Automatic Differentiation"
image: /assets/images/bias-var.png
twitter_text: "A Hands-on Introduction to Automatic Differentiation"
tags: [DeepLearning, MachineLearning, Math]
---

In this post we take a detour from the [introductory series on machine learning theory](/ml-theory-pt1), but it's a little detour; as this post shares the same curious and investigative spirit of the series in trying to take a look under the hood at the core ideas and concepts working within our practice.

If you have any experience with machine learning or its recently trending sub-discipline of deep learning, you probably know that we usually use gradient descent and its variants in the process of training our models and neural networks. From the name of the method, it's clear that the process involves computing the gradient of a function (aka its derivative with respect to each of its components) in order to figure out where to move our parameters and weights in a way that would optimize the model. In the early days, and you might have caught a glimpse of that if you ever did [Andrew Ng's MOOC](https://www.coursera.org/learn/machine-learning) or went through [Michael Nielsen's book *Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/), we used to code the gradients of the models by hand. However, as the models got more complex this became more cumbersome and tedious! This is where the modern computation libraries shone!

Libraries like [Theano](http://deeplearning.net/software/theano/), [Torch](http://torch.ch/), [Tensorflow](https://www.tensorflow.org/), [CNTK](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/) and others allow us to just write our math and they will automatically provide us with the gradients we want, which allowed the researcher to rapidly deploy and test more and more complex architectures. These libraries are able to "automagically" obtain the gradient via a technique called *Automatic Differentiation*. This post is to demystify the technique of its "magic"!

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
