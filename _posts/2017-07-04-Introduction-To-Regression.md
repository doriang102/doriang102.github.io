
In this post, we will discuss Linear Regression. We will go over, in detail, the assumptions made for this model with some concrete examples. After this we will discuss over fitting and the various reguarlization methods used in practice.

## Linear Regression

Most people are familiar with ordinary least squares. Given a collection of observations $$\{(y_i, \mathbf x_i)\}$$, where $$\mathbf x_i \in \mathbb{R}^k$$ are the features in our model, and $$\{y_i\}$$ are the observed dependent variables, which we would like to model with $$\mathbf x_i$$, we seek to find the $$\mathbf \beta \in \mathbb{R}^k$$ which minimizes

$$ \min_{\beta} \frac{1}{N} \sum_{i=1}^N(y_i - \beta \cdot \mathbf x_i)^2.$$

Formally speaking, we are modeling our dependent variable $$Y$$ as a linear function of the features $$X$$ with some error. In other words,

$$ Y - \beta \cdot \mathbf X \sim \epsilon(\beta)$$

where $$\epsilon(\beta)$$ is an error term which we would like to minimize. Less formally, the optimal $$\epsilon$$ is a random variable which will have mean zero and some unknown variance.

In fact, for any kind of regression problem, we week to find $$f: \mathbf{X} \mapsto Y$$ such that

$$ Y - f(\mathbb{X}) \sim \epsilon(\beta).$$


This seems pretty straightforward, right? The inquisitive may wonder the following: Why do we use the $$L^2$$ norm - is there a reason or advantage over using some other $$L^p$$ space? In fact, why not replace $$L^2$$ with any other appropriate metric $$(y,\mathbf x) \mapsto d(x,\mathbf x)$$?


