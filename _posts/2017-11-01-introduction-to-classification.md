(Under Construction)

# What is classification?


# Maximizing Likelihood


# Regularization

# Classification Evaluation

# Precision vs. Recall

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
