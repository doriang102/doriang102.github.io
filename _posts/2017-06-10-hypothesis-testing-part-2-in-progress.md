In this follow up, we look more carefully at the sample size necessary to make statistically valid
inferences from observations. 

Recall the Berry-Esseen theorem which, under certain assumptions, gives a rate of convergence to a normal distribution. In particular, if we have

$$ \mathbb{E}(|X_1|^3) := \rho < +\infty $$

then it follows that

$$\left|F_{n}(x)-\Phi (x)\right| \leq \frac{C\rho}{\sigma^3 \sqrt{n}},$$


Let's first consider a simple example - if we flip a coin $$n$$ times, and each time we obtain a heads, how many iterations are necessary before we are $$95%$$ sure that the coin is not fair? ie. $$P(X \leq \alpha) = 0$$?

Let's recall from the last post that we can write the posterior distribution on the coin bias as

$$ f(p | n) = \frac{p^nF(p)}{\int_0^1 p^n F(p) dp}.$$

Here we will *no longer assume that $$F \equiv 1$$*. We do asssume, however, that $$ C > f > 0 $$ uniformly in $$[0,1]$$.

Our goal is to understand the convergence rates to the ground truth in terms of the prior. These estimates are not ideally optimized, since they were done quickly on my own, as I was unable to find any good literature on this particular subject from an analyltical point of view (please email me if you know of some!). 

Let's consider the cumulative distribution function, which we denote as $$\Phi_n(p)$$:

$$ \Phi_n(x) = \frac{\int_0^x p^n f(p) dp }{\int_0^1 p^n f(p) dp}. $$

A simple application of Holder's inequality yields:

$$ \int_0^x p^n f(p) dp \leq \frac{x^{n+1}}{n+1}\|f\|_{L^{\infty}([0,x])}   $$

Using our bounds, we obtain

$$ \int_0^1 p^n f(p) dp \geq \frac{1}{n+1} \min_p f(p)  $$

Combining the above two inequalities, we obtain:

$$ |\Phi_n(x)| \leq \frac{\max_p f}{\min_p f} x^{n+1} $$

While this is a modest estimate, let's note the following:

- When $$F \equiv 1$$, we get back $$x^{n+1}$$ which is what we obtain from integrating (so it's tight in some regard)
- For any given $$\alpha \in (0,1)$$ we have 

$$ |\Phi_n(\alpha)| \leq  \frac{\max_p f}{\min_p f} \alpha^{n+1}. $$

Thus we get an exponential rate of convergence to the cumulative distribution of the dirac measure centered at $$p=1$$. 

Let's make our hypothesis:

**Hypothesis:** The coin that you're flipping gives heads more than it does tails. 

For this hypothesis, we set $$\alpha = \frac{1}{2}$$ and obtain 


$$ P(X_n \leq \alpha) = \Phi_n(\alpha) \leq  \frac{\max_p f}{\min_p f} \left(\frac{1}{2}\right)^{n+1}. $$


Thus our convergence rate is affected by the maximum and minimum of $$f$$. This inequality suggests that the best case scenario is a constant prior on the data. This of course can't be completely true, since if we guessed right the first time, our algorithm should converge instantly!


### The general case

Let's assume here that $$m = n$$.

$$ \Phi_{n,m}(x) = \frac{\int_0^x p^n(1-p)^m f(p) dp }{\int_0^1 p^n (1-p)^mf(p) dp}. $$

We start off with the following identity:

$$ \Phi_{n,m}(x) = \sum_{k=m}^{n+m} x^k (1-x)^k {n+m \choose k}. $$

Next using Stirling's estimates, we have

$$ \left(\frac{m+n}{k}\right)^k \leq { m+n \choose k } \leq \left(e\frac{m+n}{k}\right)^k.$$
