# Markov Decision Processes


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
 
