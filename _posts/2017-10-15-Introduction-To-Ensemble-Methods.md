
So far all of the models we have considered have been *parametric* - ie. there is some underlying distribution with parameters $$\theta$$ which we wish to learn. In this post, we will introduce Ensemble Methods which are non-parametric. In particular, we will cover Decision Trees and Random Forests. We will focus first on the case of classification. 

# Decision Trees

## Constructing one tree

Let's consider data $$(\mathbf X, \mathbf y)$$ where $$y \in \{0,1\}$$. For classification, we seek a rule $$\mathbf x \mapsto p(\mathbf x)$$ which maximizes:

$$ \mathcal{L}(p) := \prod_{k=1}^N p(x_i)^{y_i} (1 - p(x_i))^{1-y_i}.$$

Taking a log and dividing by $$N$$ we have:

$$ \mathcal{Q}(p) := \frac{1}{N}\sum_{k=1}^N  y_i \log p(x_i) + (1-y_i)\log (1-p(x_i)).$$

If we assume that $$x \mapsto p(x)$$ is piecewise constant on regions $$S \subset X$$ with value $$p(s)$$, then notice that

$$ \sum_{k=1}^N \frac{y_i}{N} \log p(s) = p(s) \log p(s).$$

If we do this over the segments of $$X$$ which componse the level sets of $$p$$, then we have

$$ \frac{1}{N}\sum_{k=1}^N  y_i \log p(x_i) = \sum_{k=1}^N p(x_k) \log p(x_k).$$

The calculation for the second term is similar, and optimizing $$p(X) \log P(X)$$ is the same as $$ (1-P(X)) \log 1 - P(X)$$. Thus if we constrain our functions to be piecewise constant, we wish to minimize the overall *Entropy*:

$$H(p) := \sum_{c=1}^M p_{c} \log p_c,$$

where $$p_c = p(Y = c)$$ and there are $$M$$ possible classes. To get a sense of this term, observe that it is concave with maximum at $$\frac{1}{2}$$ and $$H(0)=H(1) = 1$$:

{% highlight ruby %}
p = np.linspace(0.001,1,1000)
H = [-x*np.log(x) for x in p]
plt.plot(p,H)
{% endhighlight %}
![](/img/entropy.png?raw=true)

To fix ideas, we are going to use the Iris data set, which is a well known data set used to classify different types of flowers based on its size properties. It's a good dataset to explain the fundamental principles of decision trees since there is a rich structure within a very small dataset.

Let's load in the data and see what it looks like:

{% highlight ruby %}
import seaborn as sns; sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g = sns.pairplot(iris, hue="species")
{% endhighlight %}
![](/img/iriscorr2.png?raw=true)


## Decision Tree Algorithm

How do we minimize $$H$$ above? In decision trees we use a forward greedy method. We will assume below our data is categorical. If it is not, then we assume we have applied some bucketing to the continous values. 


First we compute the total entropy:

### Step 1: Compute the total entropy of $$Y$$.

$$ H(P) = \sum_{c=1}^M p(Y=c) \log p(Y=c).$$

This tells us how pure the classes are. In the Iris dataset above, we have $$M=3$$ classes. Let's compute this manually for Iris:
{% highlight ruby %}
def entropy(y):
    classes = np.unique(y)
    size_classes = [len(y[y==c]) for c in classes]
    N = len(y)
    entropy = -np.sum([(s/N) * np.log(s/N) for s in size_classes])
    return entropy
entropy(iris['species'])
{% endhighlight %}

{% highlight ruby %}
Output:
1.0986122886681096
{% endhighlight %}

### Step 2: For each feature $$X_j$$, compute conditional Entropy. 

Let $$x_j^k$$ be the values (or levels) that $$X_j$$ takes on. 

$$ H(P | X_j) = \sum_{k=1}^{K_j} \sum_{c=1}^M p(Y=c | X_j=x_j^k) \log p(Y=c | X_j=x_j^k).$$

We define the **Information Gain** of $$X_j$$ as:

$$ H(Y) - H(P | X_j). $$

First we break our continous features into quantiles:
{% highlight ruby %}
df_cat = pd.DataFrame(columns=iris.columns,index=range(len(iris)))
for feature in iris.columns.values[:-1]:
    df_cat[feature]=pd.qcut(iris[feature],4,range(4))

df_cat['species']=iris['species']
{% endhighlight %}

Next we will cycle through each feature and add up the entropies from each split to see which one is the best.

{% highlight ruby %}
entropy_total = 0.0
entropies=[]
for feature in iris.columns.values[0:-1]:
    entropies=[]
    vals=np.unique(df_cat[feature].values)
    entropy_total = 0.0
    for t in np.unique(df_cat[feature].values):
        entropy_total = 0.0
        df_split1 = df_cat[df_cat[feature] <= t]
        df_split2 = df_cat[df_cat[feature] > t]

        if len(df_split1)>0:
            entropy_total = entropy_total + entropy(df_split1[feature])
        if len(df_split2)>0:
            entropy_total = entropy_total + entropy(df_split2[feature])
        entropies.append(entropy_total)

    plt.plot(vals,entropies,label=feature)
    plt.legend()
    plt.savefig(path + "entropy_iris_compare.png")
{% endhighlight %}
![](/img/entropy_iris_compare.png?raw=true)

Printing the raw output we have:

{% highlight ruby %}
Feature  sepal_length
[1.0972811987124906, 1.3859818285596666, 1.0964752493540126, 1.3839038058416437]
Feature  sepal_width
[1.0944710658789472, 1.3720185953928703, 1.0813741858392665, 1.3732786055462978]
Feature  petal_length
[1.0916813968397496, 1.366834096362366, 1.0879487948449884, 1.3765642287411282]
Feature  petal_width
[1.0975084815466807, 1.383434855088411, 1.0976531947932993, 1.3840689647011553]

{% endhighlight %}


Notice that it seems that the lowest entropy split (or highest information gain) is for the `petal_length` feature at the 25% threshold. Thus `petal_length` is our top feature. 

### Step 3: Pick $$X_*^1$$ to be the feature which has largeest information gain and split. 

We perform Step 2 for every $$X_j$$, and choose our first feature be the solution of

$$ X_*^1 := \textrm{argmax}_{j} H(Y) - H(P | X_j). $$

Now split on $$X_*^1$$ to be the optimal $$x_j^i$$. In our case, the top feature was `petal_length`. Putting this into `scikit-learn`, we see it obtains the same result:

{% highlight ruby %}
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier(max_depth=1)
clf = clf.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph 
{% endhighlight %}
![](/img/iris_dt1.png?raw=true)

### Step 4: Split on the above attribute and continue recursively.

Define 

$$Y_1 := Y \lvert X_j = c_j^{*}$$ and $$Y_2 = Y \lvert X_j \neq c_j^*$$ 

in the categorical case and 

$$Y_1 := Y \lvert X_j >= c_j^{*}$$ and $$Y_2 = Y \lvert X_j < c_j^*$$ 

in the continous case. Then repeat Step 2 with the new variables $$Y_1$$ and $$Y_2$$ recursively. Let's just do one split to demonstrate how it works. Our best split was `petal_width = 0` and `petal_width > 0`:

{% highlight ruby %}
entropy(df_cat[df_cat['petal_length']==0]['species'])
Output:
0.0
{% endhighlight %}

Thus we have a pure branch on the second split! 


### Final decision tree code

Let's now use `sklearn` to make the decision tree and plot it.

{% highlight ruby %}
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
{% endhighlight %}
![](/img/iris_dt.png?raw=true)

Notice how the decision tree found the same thing we did manually? `petal_length` is the top feature with a left split that has pure class. 

# Ensemble Methods

We will go over two main methods here - bagging methods, and boosting methods. Bagging methods take multiple trees or *weak learners*, trained in parallele, then combine the result. While boosting iterates and corrects the errors on the residuals in a recursive manner. 

![](/img/ensembling.png?raw=true)


## Random Forests

Random Forests are a simple but powerful extension of decision trees which help to prevent overfitting. One simply chooses random subsets of $$\tilde K < K$$ features where $$K$$ is the number of features. You then you train $$T$$ decision trees and average the results over the trees (you can also take the mode or median if you wish). To do variable importance you simply average out over all of the trees as well. 

![](/img/rfpic.png?raw=true)

In other words, our prediction would be 

$$p_{RF}(y | x) = \frac{1}{K} \sum_{k=1}^K p_K (y \vert x),$$

where $$p_K$$ is an individual tree and $$p_{RF}$$ is the final tree based on averaging over the decision trees. Note that we can also take the mode or median instead of the average here.

The main advantages of Random Forests are:

- Not prone to over fitting due to the averaging of many trees. 
- Easy to implement 'off the shelf'.
- Easy to parallelize since the decision trees do not need to exchange information when being constructed. 


## Gradient Boosting


There is a lot of confusing notation and discussion around gradient boosting, when ultimately it is just the gradient flow in a functional space (ie. $$L^2$$). 

### Continous case

For example, consider the mean squared loss:

$$\mathcal{L}(y,f) : = \frac{1}{N} \sum_{i=1}^N (y_i - f(x_i))^.$$

We can consider the $$L^2$$ gradient flow:

$$ \partial_t f = - \nabla_{L^2} \mathcal{L}(y,f) = - \frac{2}{N} \sum_{i=1}^N (y_i - f(x_i)). $$

As with any ODE or PDE, we start with some initial condition, say $$f_0(x) = \frac{1}{N} \sum_i y_i$$, the average. How does this evolve under the above equation?

Since the above holds for all points $$x_j$$, we can write

$$ \partial_t \frac{1}{N} \sum_i (f(x_i) - y_i) = - \nabla_{L^2} \mathcal{L}(y,f) = - \frac{2}{N} \sum_{i=1}^N (y_i - f(x_i)). $$

Setting $$Z_N(t) := \frac{1}{N} \sum_i (f(x_i) - y_i)$$ and using the standard theory of ODE, we conclude that

$$Z_N(t) \sim Z_N(0) e^{-Mt},$$

where $$M$$ depends only on the Hessian of $$\mathcal{L}$$ (constant in this case). Thus if we iterate far enough, we will always converge to every point - ie. over fit! In the machine learning setting, we need to account for this by introducing regularization on the learning rate and depth of the iterations. 

### Discrete case

In the discrete case, we solve:

$$ \hat y^{t} = \hat y^{t-1} + \textrm{argmin}_h \sum_{i=1}^N\mathcal{L} (y_i, \hat y_i^{t-1} + h(x_i)). $$

Since this isn't computationally feasible, we iterate the gradient flow above with a learning rate $$\alpha$$:

$$\hat y_j^{t} = \hat y_j^{t-1} - \alpha_t \sum_{i=1}^N \nabla \mathcal{L}_{\hat y_i^{t-1}} (y_i \hat y_i^{t-1}),$$

where $$\alpha_t$$ is solved via line search. 

## Gradient Boosted Trees

So why can't we just solve the steepest descent problem? The main issue is explaiend well in *The Elements of Statistical Learning* by Hastie et al.

*If minimizing the loss of the training data were the only goal, steepest descent would be the prefered strategy. The gradient is trivial to calculate for any differentiable loss function $$\mathcal{L}(y,f(x)$$ ...Unfortunately the gradient is only defined at training points $$x_i$$, whewreas our goal is to generalize $$\hat y^t$$ to new data not represented in the training set. *

Recall that our goal is to construct an estimator, $$\hat y$$ for y which is a collection of decision trees. The Random Forest method provided one way of doing this - simply construct many decision trees in parallel, then average the results out. A method which focucses on reducing the errors made by the previous decision trees is *Gradient Boosted Decision Trees*.

Let's imagine we add a decision tree at each step:

$$
\begin{split}\hat{y}_i^{(0)} &= 0\\
\hat{y}_i^{(1)} &= f_1(x_i) = \hat{y}_i^{(0)} + f_1(x_i)\\
\hat{y}_i^{(2)} &= f_1(x_i) + f_2(x_i)= \hat{y}_i^{(1)} + f_2(x_i)\\
&\dots\\
\hat{y}_i^{(t)} &= \sum_{k=1}^t f_k(x_i)= \hat{y}_i^{(t-1)} + f_t(x_i)
\end{split}
$$

Then how do we choose the best tree?

$$
\begin{split}\text{obj}^{(t)} & = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{i=1}^t\Omega(f_i) \\
          & = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) + constant
\end{split}
$$


### XGBoost

This implementation is slightly different than the usual so we cover it seperately. 

We perform a Taylor expansion of the above and sum over all $$i$$:

$$
\mathcal{L}^{(t)}\approx \sum_{i=1}^n \underbrace{\ell(y_i,\hat{y}_i^{(t-1)})}_{\text{constant}}+\underbrace{g_i}_{\text{constant}}f_t(\mathbf{x}_i)+\frac{1}{2}\underbrace{h_i}_{\text{constant}}f_t^2(\mathbf{x}_i)+\Omega(f_t),$$

where 

$$
\begin{split}g_i &= \partial_{\hat{y}_i^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})\\
h_i &= \partial_{\hat{y}_i^{(t-1)}}^2 l(y_i, \hat{y}_i^{(t-1)})
\end{split}.
$$

Thus at each time step $$t$$, we wish to find $$f_t(x)$$ which minimizes:

$$
Q_t(f) := \sum_{i=1}^n [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t).
$$

Recalling that we are building a decision tree though, we wish to make the ansatz that $$f_t(x) = w_{q(x)}$$ where $$q: \mathbb{R}^d \to \{1,2,\cdots,T\}$$ maps each data point $$x_i$$ to one of the $$T$$ leaves. 

Substituting $$ f_t(x) = w_{q(x_)}$$ into $$Q_t(f)$$ we have


$$
Q_t(w(q)) = \sum_{i=1}^n [g_i w_{q(x_i)} + \frac{1}{2} h_i w_{q(x_i)}^2 ] + \Omega(w,q).
$$

We define our regularization term, as in XGBoost to be

$$
\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2.
$$

We can write $$Q_t$$ as 

$$
 \sum^T_{j=1} [(\sum_{i\in I_j} g_i) w_j + \frac{1}{2} (\sum_{i\in I_j} h_i + \lambda) w_j^2 ] + \gamma T
$$

where $$I_j = \{ i  \lvert q(x_i) = j\}$$ and $$j$$ denotes the leaf. Now let

$$ G_j = \sum_{i\in I_j} g_i$$
$$ H_j = \sum_{i\in I_j} h_i$$

and we can write the above loss as 

$$
 \sum^T_{j=1} [G_j w_j + \frac{1}{2} (H_j + \lambda) w_j^2 ] + \gamma T.
$$

Solving for the optimal weights $$w_j$$ for fixed struture, and substituting back into the expression, we obtain:

$$
\begin{split}w_j^\ast = -\frac{G_j}{H_j+\lambda}\\
\text{obj}^\ast = -\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j+\lambda} + \gamma T
\end{split}

$$

which can be optimized for $$G_j$$ and $$H_j$$ by recursive splitting. 
