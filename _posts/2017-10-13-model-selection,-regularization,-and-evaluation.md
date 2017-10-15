
*Under Construction*

Recall that in the last post, our aim was, given $$y \in \mathbb{R}^n$$ and our features $$X \in \mathbb{R}^{n \times k}$$ to find $$\beta \in \mathbb{R}^k$$ which minimized:

$$ \min_{\beta} \frac{1}{N} \sum_{i=1}^N(y_i - \beta \cdot \mathbf x_i)^2.$$

We will explore the idea of over fitting and model selection in this section and see the benefits and drawbacks of various methods of parameter penalizaiton. 

# Regularized Linear Regression


Let's say that our data points satisfy the following:

$$ \mathbf{y} = 10\mathbf{x_0} + \epsilon, $$

where $$\epsilon \sim \mathcal{N}(0,1)$$. 

We have $$\mathbb{R}^k$$ features, so we can actually solve:

$$ \mathbf{X}_k \beta =  \mathbf{y}, $$

for a unique $$\beta \in \mathbb{R}^k$$ where $$\mathbf{X}_k$$ is the first $$k$$ rows of $$X$$. *But wait!* Doesn't that mean we can pick up not just $$y$$, but the noise terms from $$\epsilon$$ as well? Yes! This is actually why we need regularization in the first place. Even though we have a collection of $$k$$ orthogonal features, if we have $$j < k$$ variables which influence $$y$$, then we can over solve the system. This all seems very abstract though, so let's construct a concrete example. 

We are going to construct an orthogonal matrix of dimension $$50$$ when $$y$$ depends on only one variable:

$$ \mathbf{y} = 10\mathbf{x_0} + \epsilon. $$

What do you think will happen if we include only 1 feature? 20? 40? 80? Let's write some code in Python to investigate. First let's construct the orthogonal matrix and make a scatter plot to see how $$y$$ depends on $$x$$:

{% highlight ruby %} 
from scipy.stats import ortho_group 
m = ortho_group.rvs(dim=50)
df_orth = pd.DataFrame(m).T


y = 10*x0 + np.random.normal(0, 1, 50)

plt.scatter(x0,y)

 {% endhighlight %}

![](/img/scatter_overfit.png?raw=true)

Now we have an orthogonal set of 50 features, but our output variable $$y$$ depends on only one of them. What happens as we include more of the features into our model? If you followed the previous discussion, you should know the answer already. 

{% highlight ruby %} 
from sklearn import datasets, linear_model

for d in range(0,80,20):
    regr = linear_model.LinearRegression(fit_intercept=False)
    X=df_orth.loc[:,0:d]
    # Train the model using the training sets
    regr.fit(X,y)

    # Make predictions using the testing set
    y_pred = regr.predict(X)
    plt.scatter(x0,y,color='b')
    plt.scatter(x0,y_pred,color='r')
    plt.show()
 {% endhighlight %}   
 
 
 ![](/img/overfit_0.png?raw=true)
 
 ![](/img/overfit_20.png?raw=true)
 
 ![](/img/overfit_40.png?raw=true)
 
 ![](/img/overfit_60.png?raw=true)


Do you see what's going on here? As we increase the number of features, we are able to solve for every point exactly, which is not the true trend of the model. 

**Question:** Why is this not ok? Isn't that a perfect model?

**Answer:** No! This will *not generalize properly to new data*, as we will see when we evaluate these models using cross validation in this section. 

First we define cross validation:

**Cross Validation:** Given features $$\mathbf X$$ and output $$\mathbf y$$, we train on a (ideally random) subset of rows of $$\mathbf X$$ and $$\mathbf y$$ which we will call $$\mathbf X_{\textrm{train}}$$ and $$\mathbf y_{\textrm{train}}$$ and evaluate the performance of the model on the remaining subsets which we call $$\mathbf X_{\textrm{test}}$$ and $$\mathbf y_{\textrm{test}}$$. 

**K-Fold Cross Validation:** This is the standard method of evaluating models. It's an extension of the above which involves splitting your data into $$K$$ different *folds*, where you train on a random subset and evaluate on the remaining portion ofthe data set. The following picture explains this more clearly than words can.

 ![](/img/crossval2.png?raw=true)

### Regularization - Penalizing the Size of Coefficients

In the example above, we see that as we increase the number of features we use, we over fit the model. But how do we quantify this and evaluate in a rigorous way? First we must discuss the notion of cross validation - evaluating your model on held out data. 

First, let's introduce the penalty term:

$$\sum_{k=1}^n (y_k- \beta \cdot \mathbf{x_k})^2 + \lambda \|\beta\|_{L^p}$$


 ![](/img/lassovsridge.png?raw=true)
 


On the left in the figure above is the $$L^1$$ norm and on the right the $$L^2$$ norm. As a result of the square shape, $$L^1$$ results in much sparser solutions (it's more likely to hit a kink than a side), and $$L^2$$ tends to spread out the error more. Before diving into this however, let's consider for now the $$L^1$$ norm and evaluate on held out data. 

Now let's try this on the data from before and see what we get. Note that this is a basic and crude example to demonstrate the method.

{% highlight ruby %} 
scores = []

alphas=[0,0.001,0.01]


for d in alphas:
    regr = linear_model.Lasso(alpha=d)
    X=df_orth
    
    # Train the model using the training sets
    regr.fit(X_train,y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X)
    scores.append(regr.score(X_test,y_test))
plt.plot(scores)

{% endhighlight %}   
  ![](/img/lasso1.png?raw=true)
 
 Let's do this properly now using `sklearn`'s `GridSearchCV` package:
 
 {% highlight ruby %}
 # Set the parameters by cross-validation
from sklearn.grid_search import GridSearchCV
tuned_parameters = [{'kernel': ['linear'], 'alpha': [0,0.001,0.01]}]
alphas=np.linspace(0,1,1000)
scores = ['precision', 'recall']

model=linear_model.Lasso()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(X_train,y_train)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)
{% endhighlight %}


### Requirement 1 - Standardization of independent and dependent variables.

We will introduce a basic example in this section to deleniate the need for regularization and how over fitting can occur by simply having too many variables. 

Cnsider the example where we have a simple rule $$ y = 2x_1 + 1 + \epsilon $$ where $$\epsilon \sim \mathcal{N}(0,1)$$, but we are seeking to learn a model with $$\mathbb{x} \in \mathbb{R}^d$$ for $$d > 1$$. More data is a good thing right? We'll see why it's not, and Linear Algebra will be our tour guide. 


Let's consider a one dimensional example:

$$\sum_{k=1}^n (y_k- \beta \cdot \mathbf{x_k} - \beta_0)^2 + \lambda \|\beta+\beta_0\|_{L^p}$$


If $$\{x_k\}$$ are mean-zero centered, then the first expression has mean 0 if we choose the correct intercept for $$\beta_0$$. But what if it isn't? Let's imagine that $$\{y_k\}$$ have mean $$\mu=M$$ for $$ M > > 1$$ an to fix ideas. What will be the best choice of $$\lambda$$? For each $$\lambda > 0$$ we will find that $$\beta_0 = M$$ and so we want to shrink $$\lambda \to 0$$. What effect does this have? It means that we can't penalize any of the information in $$\beta$$ and thus you will over fit.  

$$ \hat y^i = \beta_0  + \beta_1 x_1^i + \beta_2 x_2^i + \cdots + \beta_k x_k^i $$




