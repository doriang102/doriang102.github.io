

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

{% highlight ruby %} 
n = 5000
x = np.random.randn(n)
s = np.random.normal(0, 2, n)
yvals = [xval**2 + np.random.normal(0,1,1)[0] for xval in x]
y = 3*x +s

fig, ax = plt.subplots()
fit = np.polyfit(x, yvals, deg=1)
ax.plot(x, fit[0] * x + fit[1], color='red')
ax.scatter(x, yvals)
{% endhighlight %}

 ![](/img/nonlinear.png?raw=true)

### Assumption 2 - The residuals, $$\epsilon_i$$, are i.i.d. 

 We assume that $$\epsilon_i := Y_i - f(\mathbf X_i)$$ are all i.i.d random variables (indepdendent, identically distributed). 
 
 Note that it sufficies to assume independence of the error terms $$\epsilon_i$$ to conclude that $$Y_i \rvert X_i$$ are independent. 
 
 Indeed, 
 
 $$ \mathbb{E}(Y \rvert X=x_i \cdot Y \rvert X = x_j) = \mathbb{E}( (\beta \cdot x_i + \epsilon_i)  (\beta \cdot x_j + \epsilon_j)).$$

Expanding, using the fact that $$x_i$$ and $$x_j$$ are deterministic so $$\mathbb{E}(x_i) = x_i$$ and $$\mathbb{E}(x_j) = x_j$$, along with independence of $$\epsilon_i$$ and $$\epsilon_j$$, so that 

$$\mathbb{E}(\epsilon_i \epsilon_j) = \mathbb{E}(\epsilon_i) \mathbb{E}(\epsilon_j)$$, 

we have


$$ \mathbb{E}(Y | X=x_i \cdot Y | X = x_j) = (\beta \cdot x_i + \mathbb{E}(\epsilon_i))(\beta \cdot x_j + \mathbb{E}(\epsilon_j)) = \mathbb{E}(Y \rvert X = x_i) \mathbb{E}(Y \rvert X=x_j).$$

{% highlight ruby %} 
n = 5000
x = np.random.randn(n)
s = np.random.normal(0, 2, n)
yvals = [3*xval + np.random.normal(0, xval**2,1)[0] for xval in x]
y = 3*x +s

fig, ax = plt.subplots()
fit = np.polyfit(x, yvals, deg=1)
ax.plot(x, fit[0] * x + fit[1], color='red')
ax.scatter(x, yvals)
{% endhighlight %}
 ![](/img/noniid.png?raw=true)
  
  


### Assumption 2 - The residuals $$\epsilon_i$$ are all normally distributed with zero mean, $$ \epsilon_i \sim \mathcal{N}(0,\sigma^2). $$
 
 Let's take an example where we have a residual which has a non-zero mean, and see how `scikit-learn` learns the coefficients. 
 
{% highlight ruby %} 
from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
n = 50000
x = np.linspace(0,1,n)
x_pd=pd.DataFrame(x,columns=['x'])
x_pd['const']=1
yvals = [xval + np.random.normal(10,2,1)[0] for xval in x]

# Train the model using the training sets
regr.fit(x_pd,yvals)

# Make predictions using the testing set
y_pred = regr.predict(x_pd)
plt.scatter(x,yvals)
plt.plot(x,x*regr.coef_[0] + regr.coef_[1],color='r')
{% endhighlight %}
 ![](/img/nonzeromean.png?raw=true)

What went wrong here? Let's check the coefficients:

{% highlight ruby %}  
 regr.coef_
{% endhighlight %}
 
 We obtain the following model:
 
 $$ \hat y = 1.03 x.$$
 
 The model got the linear coeffficient correct, but *`scikit-learn` assumes that the data has been mean centered before training by default*. If we want to correct this issue, we can add `fit_intercept=False`:
 
  {% highlight ruby %} 
 regr = linear_model.LinearRegression(fit_intercept=False)
  {% endhighlight %}
  
  Then we obtain:
   ![](/img/zeromean.png?raw=true)


### Assumption 3 (not technically necessary) - The matrix $$\mathbf{X^TX}$$ has full rank. 

Using Calculus, we obtain the general solution when $$X^TX$$ is invertible:

$$ \hat \beta = (X^TX)^{-1} X^T y.$$

When it's not, we can still minimize the $$L^2$$ norm of the residuals, but we lack stability. To see why, a simple computation shows that:

$$ \frac{d^2}{d\beta^2} \sum_i (y_i - \mathbf{\beta} \cdot \mathbf{x_i})^2 = \frac{2}{N} X^TX.$$

From what we know about linear algebra, this matrix is positive semi-definite and symmetric. It is positive definite precisely when the columns of $$X$$ are linearly independent (ie. the features aren't correlated). Let's simulate two correlated features with noise $$x_1$$ and $$x_2$$ with a well defined linear trend:

$$ y = x_1 + \epsilon, $$

where $$\epsilon \sim \mathcal{N}(0,0.01)$$ and see how stable the coefficients are:

  {% highlight ruby %} 
from numpy import linalg as LA
n=10000
x1 = np.linspace(0,0.01,n)


k = np.random.normal(0,0.01,n)/np.linalg.norm(k)**2
s = np.random.normal(0, 0.001, n)

x2 = 100*np.linspace(0,0.01,n)
x2 -= x1.dot(k) * k / np.linalg.norm(k)**2
x1=k
df=pd.DataFrame({'x1':x1,'x2':x2})

y = x1 + np.random.normal(0, 0.01, n)
coefs1=[]
coefs2=[]
scores_perp=[]
for i in range(0,100):
    y = x1 + np.random.normal(0, 0.001, n)
    regr = linear_model.LinearRegression()
    x=df
    # Train the model 
    regr.fit(x,y)
    coefs1.append(regr.coef_[0])
    coefs2.append(regr.coef_[1]*100)
    scores_perp.append(regr.score(x,y))
    
plt.figure(figsize=(8,5))
plt.plot(coefs1,label='x1')
plt.plot(coefs2,label='x2')
plt.legend()
plt.show()
{% endhighlight %}
![](/img/perpcoefs.png?raw=true)

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

$$ \hat y^i = \beta_0  + \beta_1 x_1^i + \beta_2 x_2^i + \cdots + \beta_k x_k^i $$

Let's say that our actual points are:

$$y_1 = 2x_1^1 + M + 10^{-3}$$

$$y_2 = 2x_2^1 + M - 10^{-3}$$

$$y_3 = 2x_3^1 + M + 2\cdot 10^{-3}$$


But we have $$\mathbb{R}^k$ features, so we can solve:

$$2x_1^1 + M + 10^{-3} = \beta_1 x_1^1 + \beta_2 x_1^2 + \beta_0$$

$$2x_2^1 + M - 10^{-3} = \beta_1 x_2^1 + \beta_2 x_2^2 + \beta_0$$

$$ y_1 = M + 2x_1 + $$
$$\begin{bmatrix}a & b\\c & d\end{bmatrix}$$


