

(Under Construction)

In this post we will dig deeper into the types of distributions that one encounters in practice. It takes a considerable amount of 
time and experience to understand and appreciate which distribution is most appropriate for the problem you're trying to solve.
While this list is by no means exhaustive, it is the list of distributions which I've personally encountered a need for:

- Bernoulli
- Binomial
-  Beta
-  Normal
-  Poisson
- General Exponential

# When to use which?


### Bernoulli

The Bernoulli distribution is probably the simplest distribution, and connected with the coin toss. If I flip a coin which has probability $$p$$ of showing up, this follows a Bernoulli disttribution.

This simple distribution is used in many frameworks for modeling such as Logistic Regression, Poisson Regression, Random Walks, Bayesian Inference (seen in prevous post on Hypothesis Testing) and many others. 

It is one of the most general probability distributions, which is often replaced with ones like Poisson or Normal in the limit of large sample size (which is rigorously justified). We define the Bernoulli distribution as


$$
   f_p(k)= 
\begin{cases}
   p,& \text{if } k=0\\
    1-p,              & k=1
\end{cases}
$$

- **Logistic Regression:** Assume that $$\log \frac{p(x)}{1-p(x)} = \beta \cdot x$$ and solve by maximum Likelihood.
- **Poisson Distribution:** Assume that there are $$Np = \lambda$$ successes on average in a time interval of length $$N$$. Then we'll see that
$$ P(X_n=k) \to \frac{\lambda^k}{k!} e^{-\lambda}.$$

### Poisson

The Poisson distribution aims to model the number of independent events which occur in a fixed time interval, assuming that the probability is *memoryless*. 

