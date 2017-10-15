
So far all of the models we have considered have been *parametric* - ie. there is some underlying distribution with parameters $$\theta$$ which we wish to learn. In this post, we will introduce Ensemble Methods which are non-parametric. In particular, we will cover Decision Trees and Random Forests. We will focus first on the case of classification. 

# Classification

Let's consider data $$(\mathbf X, \mathbf y)$$ where $$y \in \{0,1\}$$. For classification, we seek a rule $$\mathbf x \mapsto p(\mathbf x)$$ which maximizes:

$$ \mathcal{L}(p) := \prod_{k=1}^N p(x_i)^{y_i} (1 - p(x_i))^{1-y_i}.$$

Taking a log and dividing by $$N$$ we have:

$$ \mathcal{Q}(p) := \frac{1}{N}\sum_{k=1}^N  y_i \log p(x_i) + (1-y_i)\log (1-p(x_i)).$$

If we assume that $$x \mapsto p(x)$$ is piecewise constant on regions $$S \subset X$$ with value $$p(s)$$, then notice that

$$ \sum_{k=1}^N \frac{y_i}{N} \log p(s) = p(s) \log p(s).$$

If we do this over the segments of $$X$$ which componse the level sets of $$p$$, then we have

$$ \frac{1}{N}\sum_{k=1}^N  y_i \log p(x_i) = \sum_{k=1}^N p(x_k) \log p(x_k).$$

# Random Forests

