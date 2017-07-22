
# Experimental Design


- **Replication**: to provide an estimate of experimental error

- **Randomization**: to ensure that this estimate is statistically valid; and

- **Local control**: to reduce experimental error by making the experiment more efficient.

- **Design structure:** to ensure that the experimental units are homogenous and represent equal samples from the same distribution.

- **Treatment structure:** to ensure that the treatment is given in a uniform way. Are a large percentage of one group receiving the treatment at a different time or in a different way?

### Replication

An essential component to a good experiment is replication. Mathematically this means that our observations are a collection of i.i.d random variables $$\{Y_i\}_{i=1}^n$$ where $$n$$ is sufficiently large so that we can read statistical significance from the results. 

### Randomization

To ensure replicability and to infer proper conclusions, proper randomization is needed. For instance, if you were testing out the efficacy of a new drug, but only tested it on people living in Iceland, you wouldn't be able to make conclusions about people living in France necessarily. 

In other words, you're assuming that you are sampling some distribution when making observations - are you sampling a large and random enough quantity to represent the distribution accurately? Randomization in the context of data science is normally sufficient to ensure proper generalization, but sometimes there are size constraints on what you are studying. 

A good example was the problem of delivering The New York Times newspaper to Starbucks restaurants - we wanted to try out several different models, but only had approximately 6k stores to choose from. When this is the case, we need to introduce local controls:

### Local Control




## Frequentist Approach

 In the last post, we discussed how to test hypotheses from frequentist and Bayesian approaches. In this post we discuss how to properly design an experiment and interpret the results, trying to emphasize pitfalls that are likely to occur. 
 
### Equality of Distributions


### Statistical Power

In this section, we look more carefully at the sample size necessary to make statistically valid
inferences from observations. 

When organizing an experiment, we must ask ourselves a few questions:

- 1) What is the hypothesis we are testing?

- 2) How big of a sample size do we need to read results in a statistically meaningful way?

- 3) What would be considered a 'positive' result?

This brings us to the topic of statistical power. Let's say for 1), we are once again testing if the red button is better than the blue button. How many people do we need to measure this? More precisely, can we estimate

$$ p(\textrm{reject } H_0 \rvert H_1 \textrm{ is true}).$$ 


In other words, what's the probability we will be able to reject the null hyothesis if the difference we observe is actually true?

Let's define some quantities:

- Let's set $$p_R - p_B = \alpha > 0$$. 

- $$N_R = N_B=N$$ are the sample sizes of the experiment (set equal for simplicity).

- $$z_{\alpha}$$ is the minimum z score we need to have the probability of observing $$\alpha$$ to be under the necessary threshold. For $$0.05$$, $$z_{\alpha}=1.645$$ for example. 


Then we have


$$ p(\textrm{reject } H_0 \rvert H_1 \textrm{ is true}) = p(z > z_{\alpha} \rvert p_R-p_B = \alpha).$$ 

From the last section, we can write this as

$$\frac{ p_R - p_B }{\sqrt{ p_R(1-p_R)/N + p_B(1-p_B)/N}} > z_{\alpha}.$$

Since $$p_R - p_B = \alpha $$ and $$\alpha$$ is generally much smaller than both $$p_R$$ or $$p_B$$, we can set $$p_R \sim p_B := p$$ in the denominator to obtain

$$ \frac{N\alpha^2}{2p(1-p)} > z_{\alpha}^2.$$

Rewriting this we obtain

$$ N \geq 2 \frac{\sigma^2 z_{\alpha}^2 }{\alpha^2},$$

where $$\sigma^2$$ is the variance of the expected baseline conversion rate. In particular, when $$z_{\alpha}=1.65$$, we have

$$ N \geq 5.44 \frac{\sigma^2 }{\alpha^2}$$

To see what different sample sizes we need, let's plot some of these curves.

First, let's write a formula to compute the p value of the difference between two conversion rates. We first define:


{% highlight ruby %}
import scipy.stats as st
def p_value(p_T,p_C,size_T,size_C):
    var = np.sqrt(p_T*(1-p_T)/size_T + p_C*(1-p_C)/size_C)
    diff = p_T-p_C
    z_score = diff/var
    p_value = st.norm.sf(abs(z_score))
    return diff, p_value
p_values=[]
diffs=[]
{% endhighlight %}

Let's then compare a list of p values of different scales:

{% highlight ruby %}
N_vals_1 = [5.44*(p*(1-p))/((0.01*p)**2) for p in p_conv]
N_vals_5 = [5.44*(p*(1-p))/((0.05*p)**2) for p in p_conv]
N_vals_10 = [5.44*(p*(1-p))/((0.10*p)**2) for p in p_conv]
N_vals_20 = [5.44*(p*(1-p))/((0.20*p)**2) for p in p_conv]

plt.figure(figsize=(8,8))
plt.xlabel('Conversion Rate')
plt.ylabel('Required Sample Size for p < 0.05')
plt.plot(p_conv,N_vals_5,label='5% lift')
plt.plot(p_conv,N_vals_10,label='10% lift')
plt.plot(p_conv,N_vals_20,label='20% lift')
plt.legend()
{% endhighlight %}

![](/img/pvalue_power.png?raw=true)

In my experience, one is generally looking for lifts of the order of $$1-2\%$$, so you can see how the sample size is incredibly important. Generally major websites can have of the order of 20 to 100 million unique cookies visit every month, and can have anywhere from 50k to 1 million actual users. 

## Normal approximations

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

