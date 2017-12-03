(Under Construction)

# What is classification?

Let's first assume that we are modeling some parameterized family of distributions with an outcome $$y \in \{0,1\}$$. In this case we wish to maximize the Likelihood function:

$$ Q(\beta) := \prod_{i=1}^N p(y_i | X_i, \beta),$$

where we assume that $$Y \sim f(\beta)$$. Since products are difficult to deal with, we take the log of this product to obtain the *Log-Likelihood*, $$L$$:

$$L(\beta) := \sum_{i=1}^n \log p(y_i | X_i, \beta). $$

If we consider a *success* to be $$y_i=1$$ then we can model this as a Binomail distribution, ie:

$$L(\beta) = \sum_{i=1}^n \log p(x_i)^{y_i} (1-p(x_i))^{1-y_i}.$$

Using properties of the logarithm, we have

$$L(\beta) = \sum_{i=1}^N y_i \log p(x_i) + (1-y_i) \log (1-p(x_i)).$$

This is our starting point for a large collection of classification models, whether we are talking about Logistic Regression or Random Forest.

### Logistic Regression
Here we assume that
$$ p(x) = \frac{1}{1 + e^{-\beta \cdot x} } .$$
### Decision Trees

# Maximizing Likelihood


# Regularization

# Classification Evaluation

# Precision vs. Recall

For classification problems, accuracy is usually not a great metric. Why? Imagine you had only $$1%$$ of your data having a positive outcome $$y = 1$$. Then simply defining $$y \equiv 0$$ would result in $$99%$$ accuracy! How do we account for this? The first way is by defining precision and recall:

**Recall:** Out of all of the positive outcomes, what percentage does your model get right? More precisely

Recall = $$ \frac{\textrm{tp}}{\textrm{tp} + \textrm{fn}}.$$

**Precision:** Out of all outcomes your model *labels* as positive, what percentage are actually positive?

Precision = $$ \frac{\textrm{tp}}{\textrm{tp} + \textrm{fp}}.$$

# Area Under the Curve (ROC)
Let's consider a concrete example where we use a very small fraction of the training data. The only purpose of this is to demonstrate the behavior of TPR and FPR as we modify the threshold for the model. We create some sample classification data:

{% highlight ruby %}
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, n_informative=5)
df_X=pd.DataFrame(X)
y_vals=np.array(y)
Xtrain = X[:9900]
Xtest = X[9900:]
ytrain = y[:9900]
ytest = y[9900:]

clf = LogisticRegression()
clf.fit(Xtrain, ytrain)
{% endhighlight %}

Now let's add the actual outcome and score to a Pandas dataframe so we can study the results:
{% highlight ruby %}
from sklearn import metrics
import pandas as pd
%matplotlib inline

df_X=pd.DataFrame(X)
df_X['score']=clf.predict_proba(df_X)[:,1]
df_X['outcome']=y_vals

{% endhighlight %}

Now that we have the dataframe, let's plot the TPR and FPR as a function of our threshold:
{% highlight ruby %}
import seaborn
import matplotlib.pyplot as plt
fprs=[]
tprs=[]
for i in range(0,11):
    threshold = float(i)/10
    print (threshold)
    fig, ax = plt.subplots()
    df_X[df_X['outcome']==1]['score'].hist(bins=20,color='b',alpha=0.5)
    df_X[df_X['outcome']==0]['score'].hist(bins=20,color='r',alpha=0.5)
    fpr = np.round(len(df_X[(df_X['outcome']==0) & (df_X['score']>threshold)])/len(df_X[df_X['outcome']==0]),2)
    tpr = np.round(len(df_X[(df_X['outcome']==1) & (df_X['score']>threshold)])/len(df_X[df_X['outcome']==1]),2)
    fprs.append(fpr)
    tprs.append(tpr)
    ax.axvspan(threshold, 1, alpha=0.5, color='g',label='Labeled Positive - FRP= ' + str(fpr)+' TPR = ' + str(tpr))
    plt.legend()
    plt.savefig("../img/roc_" + str(i) + ".png")
    plt.show()
{% endhighlight %}

 ![](/img/roc_0.png?raw=true)
 ![](/img/roc_2.png?raw=true)
   ![](/img/roc_4.png?raw=true)
    ![](/img/roc_6.png?raw=true)
     ![](/img/roc_8.png?raw=true)
      ![](/img/roc_9.png?raw=true)

Now that we've calculated all of the tue positive rates (TPR) and false positive rates (FPR), let's plot them:

![](/img/roc_final.png?raw=true)
