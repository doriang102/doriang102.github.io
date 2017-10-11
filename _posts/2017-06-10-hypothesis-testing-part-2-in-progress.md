
# Experimental Design


- **Replication**: to provide an estimate of experimental error

- **Randomization**: to ensure that this estimate is statistically valid; and

- **Local control**: to reduce experimental error by making the experiment more efficient.

- **Design structure:** to ensure that the experimental units are homogenous and represent equal samples from the same distribution.

- **Treatment structure:** to ensure that the treatment is given in a uniform way. Are a large percentage of one group receiving the treatment at a different time or in a different way?

An example of bad experimental design is shown here:

![](/img/badexp.png?raw=true)

In the above, we see that the treatment and control groups have significantly different browsing characteristics. Thus their behavior is fundamentally different, and we can't expect to interpret the result of a uniform treatment on both groups. This lacks proper **randomization**. It could be fixed by local control - ie. sampling one distribution to match the other one so that the parameters in question are comparable. 


### Replication

An essential component to a good experiment is replication. Mathematically this means that our observations are a collection of i.i.d random variables $$\{Y_i\}_{i=1}^n$$ where $$n$$ is sufficiently large so that we can read statistical significance from the results. 

### Randomization

To ensure replicability and to infer proper conclusions, proper randomization is needed. For instance, if you were testing out the efficacy of a new drug, but only tested it on people living in Iceland, you wouldn't be able to make conclusions about people living in France necessarily. 

In other words, you're assuming that you are sampling some distribution when making observations - are you sampling a large and random enough quantity to represent the distribution accurately? Randomization in the context of data science is normally sufficient to ensure proper generalization, but sometimes there are size constraints on what you are studying. 

A good example was the problem of delivering The New York Times newspaper to Starbucks restaurants - we wanted to try out several different models, but only had approximately 6k stores to choose from. When this is the case, we need to introduce local controls:

### Local Control

### Equality of Distributions


## Frequentist Approach

 In the last post, we discussed how to test hypotheses from frequentist and Bayesian approaches. In this post we discuss how to properly design an experiment and interpret the results, trying to emphasize pitfalls that are likely to occur. 
 

Let's first define some quantities.

### $$\alpha$$ value

The threshold you choose your cut off at. More precisely,

$$ p(\textrm{ reject }  H_0  \rvert H_0 \textrm{ is true } ),$$

which is also known as a Type I error. For example, a significance level of 0.05 indicates a 5% risk of concluding that a difference exists when there is no actual difference.

### P Value

This is the probability of observing larger than or equal to the observed value when the null hypothesis is true, ie.

$$ p(Y \geq Y_{\textrm{obs}} \rvert H_0 \textrm{ is true } )$$

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


## Bayesian Statistical Power

We saw in the last post that 

$$ p^n (1-p)^{N-n} \to \delta_{\frac{n}{N}}(p),$$
in the sense of distributions. How do we infer confidence in the Bayesian setting? Let's now assume that we have
$$ \beta = \frac{n}{N} $$ successes.

With what confidence can we infer that the button has bias $\beta$?

Let's define 

$$H_1$$ - The hypothesis that $$p_1 = p_2 $$. 

$$H_2$$ - The hypothesis that $$p_1 > p_2 $$. 

We can first consider the Baye's ratio:

$$ \frac{p(D \lvert H_1)}{p(D \lvert H_2)}.$$ We can write this as 

$$ \frac{p(D \lvert H_1)}{p(D \lvert H_2)} = \frac{(\beta+\epsilon)^n (1- (\beta+\epsilon))^{N-n} }{ \beta^n (1- \beta)^{N-n}}.$$

**Lemma 1:**  *Let $$ \beta = \frac{n}{N}$$. Then it holds that*

$$\frac{p(D \lvert H_1)}{p(D \lvert H_2)} = e^{\frac{-N c_{\beta,\epsilon} \epsilon^2}{\sigma_{\beta^2}}},$$

_where_  $$\sigma_{\beta} = \beta (1- \beta) $$ _and $$c_{\beta,\epsilon}$$ *is a constant satisfying*

$$ \frac{1}{4} \leq c_{\beta, \epsilon} \leq 1. $$



**Proof:**

Let's take the log of the ratio above. Then we have


$$\frac{1}{N} \log \left(\frac{p(D \lvert H_1)}{p(D \lvert H_2)}\right) = \beta \log \left(1+ \frac{\epsilon}{\beta}\right) + (1-\beta) \log \left ( 1 - \frac{\epsilon}{1-\beta}\right).$$

We expand the logs to second order via Taylor series with remainder to obtain:

$$ \frac{1}{N}\log \left(\frac{p(D \lvert H_1)}{p(D \lvert H_2)}\right) = -\frac{c_{\epsilon,\beta} \epsilon^2}{2\beta} - \frac{\tilde c_{\epsilon,\beta}\epsilon^2}{2(1-\beta)} .$$

Since we have

$$ \frac{d^2}{dx^2} \log (1 + x) = - \frac{1}{1+x^2}, $$

by Taylor's formula we can bound $$c_{\epsilon,\beta}$$ and $$\tilde c_{\epsilon,\beta}$$ by:

$$ \frac{1}{4} \leq c_{\alpha, \beta}, \tilde c_{\alpha, \beta} \leq 1.$$

Thus we have

$$ -\frac{c_{\epsilon,\beta} \epsilon^2}{2\beta} - \frac{\tilde c_{\epsilon,\beta}\epsilon^2}{2(1-\beta)} = -\left(\beta c_{\epsilon,\beta} + (1-\beta) \tilde c_{\epsilon,\beta}\right) \frac{\epsilon^2}{2\sigma_{\beta}^2}.$$




Simplifying, and changing the constant $$c_{\beta, \epsilon}$$, the right side becomes

 $$-\frac{c_{\beta,\epsilon} \epsilon^2}{2\sigma_{\beta}^2}.$$
 
 Taking the exponential completes the argument. 

**Q.E.D.**


Thus we have immediately from the above the following Corollary. 

**Corollary 1:**  _Let_ 

$$R := \frac{p(D \lvert H_1)}{p(D \lvert H_2)} \leq e^{-\frac{N \epsilon^2}{8\sigma_{\beta}^2}}.$$

_Then we have_

$$ N \geq \frac{8\sigma_{\beta}^2}{\epsilon^2} \log \frac{1}{R}.$$


What we really want to estimate is 

$$ \frac{\int\int_{p_1 > p_2} p_1^{n_1} (1-p_1)^{N-{n_1}} p_2^{n_2} (1-p_2)^{N-n_2} dp_1dp_2}{\int \int p_1^{n_1} (1-p_1)^{N-n_1} p_2^{n_2} (1-p_2)^{N-n_2} dp_1dp_2}$$

We can rewrite the numerator as

$$ \int_0^1 p_2^{n_2} (1-p_2)^{N-n_2} \int_0^{p_2} s^{n_1} (1-s)^{N-n_1} ds  dp_2.$$

Separating the denominator out, we have


$$ \frac{1}{B(n_2+1,N-n_2-1)}\int_0^1 \Phi_{n_1}(p_1) p_2^{n_2} (1-p_2)^{N-n_2} dp_2.$$

Let's start off with an easier Lemma though. 

**Lemma 2:** _Let_ 

$$ \Phi_n(x) := \frac{\int_0^x p^n (1-p)^{N-n}}{\int_0^1 p^n (1-p)^{N-n}}. $$

Then we have for $$|x - \beta| > \alpha$$,

$$ \lvert\Phi_n(x) - \mathbf{1}[\beta,1)\rvert| \leq 2e^{-\frac{N\alpha^2}{\sigma_{\beta}^2}}), $$

and $$\Phi_n(\beta) \to 1/2$$ exponentially fast. 


**Proof:**

Courtesy of Lemma 1, we can write the function as 

$$ \Phi_n(x) := \frac{\int_0^x e^{-N\frac{c_{\epsilon,\beta} (p-\beta)^2}{\sigma_{\beta}^2}}dp}{\int_0^1 e^{-N\frac{c_{\epsilon,\beta} (p-\beta)^2}{\sigma_{\beta}^2}}dp}. $$

Let's introduce the change of variables

$$ y =  \frac{\sqrt{N c_{\epsilon,\beta}} (p-\beta)}{\sigma_{\beta}}.$$

Then we have

$$  \Phi_n(x) = \frac{\int_{-\frac{\sqrt{Nc_{\epsilon,\beta}}\beta}{\sigma_{\beta}}}^{\frac{\sqrt{Nc_{\epsilon}}(x-\beta)}{\sigma_{\beta}}} e^{-y^2} dy}{\int_{-\frac{\sqrt{Nc_{\epsilon,\beta}}\beta}{\sigma_{\beta}}}^{\frac{\sqrt{Nc_{\epsilon}}(1-\beta)}{\sigma_{\beta}}} e^{-y^2} dy}$$

Using the stanard Calculus trick of changing to polar coordinates, we can bound the Guassian integral:

$$\sqrt{2\pi} \sqrt{ 1 - e^{-a^2}} \leq \int_{-a}^a e^{-y^2} dy \leq \sqrt{2\pi} \sqrt{1 - e^{2a^2}},$$

$$\int_{a}^{+\infty} e^{-y^2} dy \leq \sqrt{2\pi} e^{-a^2}.$$

Using this we obtain the bound for $$x < \beta - \alpha $$:

$$\Phi_n(x) \leq \frac{ e^{-N c_{\epsilon}^2 \alpha^2/\sigma_{\beta}^2}}{\sqrt{1 - e^{-N c_{\epsilon,\beta}^2\beta^2/\sigma_{\beta}^2}}} \leq 2e^{-N c_{\epsilon}^2 \alpha^2/\sigma_{\beta}^2}$$

For $$ x > \alpha + \beta$$,

$$ \Phi_n(x) \geq \frac{ \sqrt{1 - e^{-N c_{\epsilon,\beta}^2 \alpha^2/\sigma_{\beta}^2}}}{\sqrt{1 - e^{-N c_{\epsilon,\beta}^2\beta^2/\sigma_{\beta}^2}}} \geq 1 - e^{-N c_{\epsilon,\beta}^2\beta^2/\sigma_{\beta}^2}.$$
