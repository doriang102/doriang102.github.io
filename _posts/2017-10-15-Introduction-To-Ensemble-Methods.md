
So far all of the models we have considered have been *parametric* - ie. there is some underlying distribution with parameters $$\theta$$ which we wish to learn. In this post, we will introduce Ensemble Methods which are non-parametric. In particular, we will cover Decision Trees and Random Forests. We will focus first on the case of classification. 

# Classification

## Optimization Problem

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

This tells us how pure the classes are. To get a s

### Step 2: For each feature $$X_j$$, compute conditional Entropy. 

Let $$x_j^k$$ be the values (or levels) that $$X_j$$ takes on. 

$$ H(P | X_j) = \sum_{k=1}^{K_j} \sum_{c=1}^M p(Y=c | X_j=x_j^k) \log p(Y=c | X_j=x_j^k).$$

We define the **Information Gain** of $$X_j$$ as:

$$ H(Y) - H(P | X_j). $$

### Step 3: Pick $$X_*^1$$ to be the feature which has largeest information gain and split. 

We perform Step 2 for every $$X_j$$, and choose our first feature be the solution of

$$ X_*^1 := \textrm{argmax}_{j} H(Y) - H(P | X_j). $$

Now split on $$X_*^1$$ to be the optimal $$x_j^i$$. 

### Step 4: Split on the above attribute and continue recursively.

Define 

$$Y_1 := Y \lvert X_j = c_j^{*}$$ and $$Y_2 = Y \lvert X_j \neq c_j^*$$ 

in the categorical case and 

$$Y_1 := Y \lvert X_j >= c_j^{*}$$ and $$Y_2 = Y \lvert X_j < c_j^*$$ 

in the continous case. Then repeat Step 2 with the new variables $$Y_1$$ and $$Y_2$$ recursively. 
# Random Forests
