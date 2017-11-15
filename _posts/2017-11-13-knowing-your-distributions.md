

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

# Applications of Each Distribution


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

Important applications:
- **Logistic Regression:** Assume that $$\log \frac{p(x)}{1-p(x)} = \beta \cdot x$$ and solve by maximum Likelihood.
- **Random Walk:** If a person has a fixed probability $$p$$ of moving right (so $$1-p$$ of moving left), then this process is called a *Random Walk*. This will have important implications when we discuss recommendation engines.

## Binomial
For this, we take $$N$$ trial of the Bernoulli distribution and obtain:

 $$ f_p^{N}(k) = {N \choose k} p^k (1-p)^{N-k}.$$
 
 The factorial takes into account the number of ways we can have $$k$$ "successes" out of $$N$$ trials. 
 
=

- **Poisson Distribution:** Assume that there are $$Np = \lambda$$ successes on average in a time interval of length $$N$$. Then we'll see that
$$ P(X_n=k) \to \frac{\lambda^k}{k!} e^{-\lambda}.$$
- **Normal Distribution:** If $$p$$ is fixed independent of $$N$$, then by the central limit theorem we have

$$f_p^{N} \to \mathcal{N}(np,np(1-p)).$$


### Poisson

The Poisson distribution aims to model the number of independent events which occur in a fixed time interval, assuming that the probability is *memoryless*. 

### Exponential

For a Poisson process, hits occur at random independent of the past, but with a known long term average rate $$\lambda$$ of hits per time unit. The Poisson distribution would let us find the probability of getting some particular number of hits.
Now, instead of looking at the number of hits, we look at the random variable L (for Lifetime), the time you have to wait for the first hit.

The probability that the waiting time is more than a given time value is 

$$P(L > t) = P(\textrm{ no hits at time t }) =  e^{-\lambda t}. $$

Then

$$ P(L \leq t) = 1 - e^{-\lambda t}. $$


 We can get the density function by taking the derivative of this:
$$ 
   f(t)= 
\begin{cases}
   \lambda e^{-\lambda t},& \text{if } t\geq 0\\
    0,              & t < 0
\end{cases} 
$$

