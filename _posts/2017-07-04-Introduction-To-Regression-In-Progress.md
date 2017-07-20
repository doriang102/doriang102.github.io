
# In progress


In this post, we will discuss Linear Regression. We will go over, in detail, the assumptions made for this model with some concrete examples. After this we will discuss over fitting and the various reguarlization methods used in practice.

## Linear Regression

Most people are familiar with ordinary least squares. Given a collection of observations $$\{(y_i, \mathbf x_i)\}$$, where $$\mathbf x_i \in \mathbb{R}^k$$ are the features in our model, and $$\{y_i\}$$ are the observed dependent variables, which we would like to model with $$\mathbf x_i$$, we seek to find the $$\mathbf \beta \in \mathbb{R}^k$$ which minimizes

$$ \min_{\beta} \frac{1}{N} \sum_{i=1}^N(y_i - \beta \cdot \mathbf x_i)^2.$$

Formally speaking, we are modeling our dependent variable $$Y$$ as a linear function of the features $$X$$ with some error. In other words,

$$ Y - \beta \cdot \mathbf X \sim \epsilon(\beta)$$

where $$\epsilon(\beta)$$ is an error term which we would like to minimize. Less formally, the optimal $$\epsilon$$ is a random variable which will have mean zero and some unknown variance.

In fact, for any kind of regression problem, we week to find $$f: \mathbf{X} \mapsto Y$$ such that

$$ Y - f(\mathbf{X}) \sim \epsilon(f).$$


This seems pretty straightforward, right? The inquisitive may wonder the following: 

- Don't we want $$ \epsilon (\beta) $$ to be as close to zero as possible? What do we mean when we call this a random variable?
- Why do we use the $$L^2$$ norm - is there a reason or advantage over using some other $$L^p$$ space? In fact, why not replace $$L^2$$ with any other appropriate metric $$(y,\mathbf x) \mapsto d(x,\mathbf x)$$? 


To answer the first question. Think of the following examples:

- Predicting the height of an individual based on age, gender and race. For a given combination (ie. (13, male, caucasian) ), we will have many possible values if we consider the entire population. There will be a fixed mean but it will vary. 
- Predicting flight delays using the airline, destination, source location and weather. There will be many different time delays for any given combination of the above variables.

Thus we can't actually expect to predict any exact value when considering stochastic values. The two main things we need to consider are:

- What distribution does $$p(y\rvert x,\beta)$$ follow? 
- What is the best choice of $$\beta$$ given this prior?


**Note:** We have assumed here that the distribution is *parametric*, ie. $$\exists \beta$$ such that the data is distributed via $$ p(y\rvert x,\beta)$$. For example 

$$p(y \rvert x, \beta) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(y-\beta \cdot x)^2}{2 \sigma^2}}.$$



### Assumption 1 - Linear relationship between dependent and indepdent variables. 

We assume that 

$$ Y = \beta \cdot \mathbf{X} + \epsilon , $$

where $$\epsilon$$ is not influenced by $$\mathbf{X}$$. 

### Assumption 2 - i.i.d of residuals $$\epsilon_i$$. 

 We assume that $$\epsilon_i := Y_i - f(\mathbf X_i)$$ are all i.i.d random variables (indepdendent, identically distributed). 
 
 Note that it sufficies to assume independence of the error terms $$\epsilon_i$$ to conclude that $$Y_i \rvert X_i$$ are independent. 
 
 Indeed, 
 
 $$ \mathbb{E}(Y \rvert X=x_i \cdot Y \rvert X = x_j) = \mathbb{E}( (\beta \cdot x_i + \epsilon_i)  (\beta \cdot x_j + \epsilon_j)).$$

Expanding, using the fact that $$x_i$$ and $$x_j$$ are deterministic so $$\mathbb{E}(x_i) = x_i$$ and $$\mathbb{E}(x_j) = x_j$$, along with independence of $$\epsilon_i$$ and $$\epsilon_j$$, so that 

$$\mathbb{E}(\epsilon_i \epsilon_j) = \mathbb{E}(\epsilon_i) \mathbb{E}(\epsilon_j)$$, 

we have


$$ \mathbb{E}(Y | X=x_i \cdot Y | X = x_j) = (\beta \cdot x_i + \mathbb{E}(\epsilon_i))(\beta \cdot x_j + \mathbb{E}(\epsilon_j)) = \mathbb{E}(Y \rvert X = x_i) \mathbb{E}(Y \rvert X=x_j).$$
 

### Assumption 2 - The residuals $$\epsilon_i$$ are all normally distributed with zero mean. 

 $$ \epsilon_i \sim \mathcal{N}(0,\sigma^2). $$


### Assumption 3 (not technically necessary) - The matrix $$\mathbf{X^TX}$$ has full rank. 



## Assumptions of Linear Regression one can violate

- Implicit independent variables (covariates):
- Lack of independence in Y:


- Outliers:
- Nonnormality:
- Variance of Y not constant:
- The correct model is not linear:
- The X variable is random, not fixed:

# Regularized Linear Regression

### Requirement 1 - Standardization of independent and dependent variables.

This is needed since we are penalizing the coefficients $$\beta$$ equally regardless of whether we use $$L^2$$ or $$L^1$$. 


Indeed, consider the example where we have a simple rule $$ y = 2x_1 + 1 + \epsilon$$ where $$\epsilon \sim \mathcal{N}(0,1)$$, but we are seeking to learn a model with $$\mathbb{x} \in \mathbb{R}^d$$ for $$d > 1$$. Clearly we can over fit this model. 

We seek to find a model:
$$ y = \beta \cdot \mathbf{x} + \beta_0 + \epsilon_i,$$
where $$ \beta_0 \in \mathbb{R}^d$$ is non-zero. More precisely we seek to minimize

$$\sum_{k=1}^n (y_k- \beta \cdot \mathbf{x_k} - \beta_0)^2 + \lambda \|\beta+\beta_0\|_{L^p}$$


If $$\{x_k\}$$ are mean-zero centered, then the first expression has mean 0 if we choose the correct intercept for $$\beta_0$$. But what if it isn't? Let's imagine that $$\{y_k\}$$ have mean $$\mu=M$$ for $$ M > > 1$$ an to fix ideas. What will be the best choice of $$\lambda$$? For each $$\lambda > 0$$ we will find that $$\beta_0 = M$$ and so we want to shrink $$\lambda \to 0$$. What effect does this have? It means that we can't penalize any of the information in $$\beta$$ and thus you will over fit.  


$$\begin{bmatrix}a & b\\c & d\end{bmatrix}$$


