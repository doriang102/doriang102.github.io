In this first blog post, I plan on discussing the detailed mathematics behind computing p values in data science, but restricted to a single hypothesis (multiple hypothesis testing will be covered later). Almost all of the expalanations I've found skip *very* important details that highlight the issues and limitations with using p values to make conclusions about the effectiveness of treatment.

## Frequentist Appraoch - p values

Let's assume that we have two buttons, a red button and a blue button. We wish to construct a proper experiment to test out which button results in higher conversions (clicks/likes, etc). 


Let $$X_i^R$$ and $$X_i^B$$ be the outcomes of the $$ith$$ observation for the red and blue buttons respectively, ie. $$X_i^{R,B} = 1$$ if the user clicked the button, and $$X_i^{R,B} = 0$$ otherwise.  Then we define the number of clicks of the red and blue buttons after $$N_R$$ and $$N_B$$ respective trials:

$$
S_N^R = \sum_{i=1}^{N_R} X_i^R
$$

$$
S_N^B = \sum_{i=1}^{N_B} X_i^B.
$$

**We make the following assumptions:**
- Both $$\{X_i^R\}$$ and $$\{X_i^B\}$$ form a collection of independent, identitically dstirbuted random variables (i.i.d). 
- Moreover, for each $$i$$, $$X_i^R$$ and $$X_i^B$$ are sampled from fixed Bernoulli distributions with means $$p_R$$ and $$p_B$$ respectively.

As a result of the **Law of Large Numbers**, we have $$\frac{1}{N_R}S_N^R \to p_R$$ and $$\frac{1}{N_B}S_N^B \to p_B$$ as $$N_R,N_B \to +\infty$$ in the sense of distributions.

The **Central Limit Theorem** tells us the next order correction term is actually normal:


$$\frac{1}{\sqrt{N_R}} \sum_{i=1}^{N_R} X_i^R \to \mathcal{N}(p_R, \sqrt{p_R(1-p_R)})$$


$$\frac{1}{\sqrt{N_B}} \sum_{i=1}^{N_B} X_i^B \to \mathcal{N}(p_B, \sqrt{p_B(1-p_B)})$$



Another way to write the above is


$$\frac{1}{N_R}\sum_{i=1}^{N_R} X_i^R - p_R + E_1 \sim \mathcal{N}\left(0, \frac{p_R(1-p_R)}{N_R}\right)$$


$$\frac{1}{N_B}\sum_{i=1}^{N_B} X_i^B - p_B + E_2 \sim \mathcal{N}\left(0, \frac{p_B(1-p_B)}{N_B}\right),$$

where $$E_1$$ and $$E_2$$ are errors that tend to $$0$$ as $$N_R, N_B \to +\infty$$. 

We've make use of the following facts:

-  We can absorb the $$\sqrt{N_R}$$ and $$\sqrt{N_B}$$ terms into the variances of the normal distributions. 
- The difference of two normally distributed random variables $$\mathcal{N}_1(\mu_1,\sigma_1)$$ and $$\mathcal{N}_2(\mu_2,\sigma_2)$$ is again a normally distributed random variable with mean $$\mu_1 - \mu_2$$ and variances $$\sigma_1^2 + \sigma_2^2$$. 

Let's plot the Binomial distributions and see how the red and blue button distributions look for fixed values:

{% highlight ruby %}
import numpy as np; np.random.seed(10)
import seaborn as sns; sns.set(color_codes=True)

# Set number of observations.
n_R=1000
n_B=1200

# Set conversion rates of observations.
p_R=0.1
p_B=0.12

# Set number of samples to take
samples=10000

# Sample from red and blue butotn given observed conversion rates. 
x_R = np.random.binomial(n_R, p_R, samples)/n_R
x_B = np.random.binomial(n_B, p_B, samples)/n_B

# Create pandas series
x_R=pd.Series(x_R)
x_B=pd.Series(x_B)

# Plot the results. 
x_B.plot(kind='kde',label='Blue Button',color='b')
x_R.plot(kind='kde',label='Red Button',color='r')
sns.distplot(x_R,kde=False,norm_hist=True)
sns.distplot(x_B,kde=False,color='r',norm_hist=True)


x_position = 0.11
plt.axvline(x_position)
plt.legend()
{% endhighlight %}
![](/img/redvsblue.png?raw=true)

These distributions *look* approximately normal which is good. Now to continue with the frequentist approach, we need to introduce the null hypothesis. 

The above difference can be re-written as:

$$\frac{\frac{1}{N_R}\sum_{i=1}^{N_R} X_i^R - \frac{1}{N_B}\sum_{i=1}^{N_B} X_i^B}{(1/\sqrt{N_R})\sqrt{  p_R(1- p_R)} +(1/\sqrt{N_B})\sqrt{  p_B(1- p_B)}} + E_4 \sim \mathcal{N}(0,1),$$

where $$E_4 \to 0$$ as $$N_R,N_B \to +\infty$$. 


**Null Hypothesis:** We assume that $$p_B = p_R$$. How probable is our observed result?


Now **we do not know $$p_R$$ or $$p_B$$, even when we assume they're equal.** 

However we have the following:

$$
\frac{1}{N_R} \sum_{i=1}^{N_R} X_i^R  = \hat p_R
$$

$$
\frac{1}{N_B} \sum_{i=1}^{N_B} X_i^B = \hat p_B,
$$

which, courtesy of the fact that that $$\hat p_R \to p_R$$ and $$\hat p_B \to p_B$$ in probability, we can replace $$p_R$$ and $$p_B$$ with $$\hat p_R$$ and $$\hat p_B$$ by absorbing the error into $$E_4$$. 


Let's now simulate this with fixed $$p_B$$ and $$p_R$$

{% highlight ruby %}
# Set number of observations.
n_R=100
n_B=120

for f in range(1,5):
    n_R = n_R*f
    n_B = n_B*f
    # Set conversion rates of observations.
    p_R=0.1
    p_B=0.12

    # Set number of samples to take
    samples=10000

    x_null=np.random.normal(0, np.sqrt(p_R*(1-p_R)/n_R) + np.sqrt(p_B*(1-p_B)/n_B), samples)
 
    # Create pandas series
    x_null=pd.Series(x_null)
    plt.xlim([-0.5,0.5])
    
    # Plot the distribution
    x_null.plot(kind='kde',label='N=' + str(n_R),color='g')
    sns.distplot(x_null,kde=False,norm_hist=True)

    # Plot the observed difference of p_B-p_R.
    x_position = 0.02
    plt.axvline(x_position)

    plt.legend()
    plt.show()
   {% endhighlight %}
![](/img/normconv1.png?raw=true)
![](/img/normconv2.png?raw=true)
![](/img/normconv3.png?raw=true)


The area to the right of the line is the probability of observing the difference we have or larger, under the null hypothesis.
In other words, it is the probability of observing any $$z \geq z_n$$ from $$\mathcal{N}(0,1)$$ where

$$
\hat z_N = \frac{\hat p_R - \hat p_B}{(1/\sqrt{N_R})\sqrt{ \hat p_R(1- \hat p_R)} +(1/\sqrt{N_B})\sqrt{ \hat p_B(1- \hat p_B)}}.
$$


This is preceily the p value, ie.

$$p = \Phi( z \geq z_n),$$


where $$\Phi$$ is a standard unit normal. 


However notice that we've just conveniently skipped over the error $$\tilde E_4$$, **which there are no currently known estimates for**. For this reason, I would take p values for Bernoulli trials with a grain of salt. 


## Bayesian Approach - Distribution on Parameters 

How how do we measure if blue is indeed better than red?

{% highlight ruby %}
# Plot the difference and shade in the probability that blue is better than red. 
x_diff = x_B - x_R
x_diff.plot(kind='kde',label='Difference',color='g')
sns.distplot(x_diff[x_diff>0],kde=False,norm_hist=True)

x_position = 0.0
plt.axvline(x_position)
{% endhighlight %}
![](/img/rawdiff.png?raw=true)



**Question:** What is the probability of observing a value equal to or larger than the above value from a normal distribution with mean 0 and variance 1? This is what a p value is.

$$P[p_1 > p_2;f_1,f_2] = \frac{\int_0^1 \int_0^1 I(p_1 > p_2) P[D_1|p_1] P[D_2|p_1] dF_1(p_1) dF_2(p_2)}{\int_0 ^1 \int_0^1 P[D_1|p_1] P[D_2|p_1] dF_1(p_1) dF_2(p_2) }$$

## Bayesian Approach


Let's define $$X_{N}$$ to be the number of heads obtained after $$N$$ flips of a coin which has bias $$p$$. Then $$X_N$$ has a distribution given by

$$ f(X_N = k | p)  = {N \choose k} p^k (1-p)^{N-k} .$$

But we really want to know what is $$ f(p | X_n = k)$$, ie. we want to learn the bias from the data. Thus we use Baye's theorem:

$$ f(p | X_N = k) =\frac{ f(X_N = k | p) f(p)}{f(X_n=k)}.$$

What is $$f(X_n=k)$$? This is summing up all possibilities over all probabilities. In other words, we have
$$ f(X_n=k) = \int_0^1 {N \choose k} p^k (1-p)^{N-k} f(p) dp.$$

Thus we have 
$$ f(p | X_N = k) =\frac{  {N \choose k} p^k (1-p)^{N-k}f(p)}{ \int_0^1 {N \choose k} p^k (1-p)^{N-k} f(p) dp}.$$

What is $$f(p)$$? This is just our prior belief about what $$p$$ is - since we don't know anything, we set it to be the *uniform distribution*, so $$f(p) \equiv 1$$. Thus we finally have

$$ f(p | X_N = k) = \frac{ p^k(1-p)^{N-k}}{\int_0^1 p^k (1-p)^{N-k} dp}.$$

For simplicity in the next sections, are going to write $$N = n + m$$ so that we have
$$ f(p | X_n = m) = \frac{ p^m(1-p)^{n}}{\int_0^1 p^m (1-p)^{n} dp}.$$

Also recall that the Beta function is defined as 

$$B(n+1,m+1) = \int_0^1 p^{n} (1-p)^m dp .$$

### Case 1: $$m = 0$$ as  $$n \to +\infty$$

In this case we have
$$ f(p | m= 0, n) = \frac{p^n}{\int_0^1 p^n dp} = (n+1)p^n .$$

It is easy to verify that the above converges in a distributional sense to $$\delta_{p=1}(p)$$. 
This works the same when $$m = o(n)$$ as $$n \to +\infty$$. We leave it as an exercise for the reader to ensure the calculations work the same. 

### Case 2: $$m = n$$ as  $$n \to +\infty$$



We begin by computing the expected value and variance of $$(p | X_n = n)$$:

$$\mathbb{E} (p | X_n = n) = \frac{B(n+2,n+1)}{B(n+1,n+1)} = \frac{n+1}{2n+3} \frac{B(n+1,n+1)}{B(n+1,n+1)} = \frac{n+1}{2n+3}.$$

$$\textrm{Var}(p | X_n=n) = \frac{1}{B(n+1,n+1)} \int_0^1 (p- \frac{n+1}{2n+3})^2 p^n (1-p)^n dp$$

$$\frac{B(n+3,n+1) - B(n+2,n+1)(n+1)/(2n+3) + (n+1)^2/(2n+3)^2 B(n+1,n+1)}{B(n+1,n+1)}.$$

Using the identity $$B(m+1,n) = \frac{m}{m+n} B(m,n)$$ repeatedly and using the fact that $$n \to + \infty$$, we have 
$$\textrm{Var}(p | X_n=n) = o(1) \textrm{ as } n \to +\infty.$$

$$\mathbb{E}(p | X_n=n) =  \frac{1}{2} +  o(1) \textrm{ as } n \to +\infty.$$


Let $$f \in C^2([0,1])$$, and let's do a Taylor expansion of $$f$$ around $$1/2$$. 
$$ f(p) = f(1/2) + f'(1/2)(p-1/2) + \frac{1}{2}f''(\xi)(p-1/2)^2,$$
where $$\xi \in [0,1/2]$$. 
Then we have
$$\int f(p) d\mathbb{P}_n(p) = f(1/2) +  f'(1/2) \left(\mathbb{E}(p | X_n = n) - 1/2\right) + \frac{1}{2} f''(\xi) \textrm{Var}(p | X_n=x)$$

which becomes

$$= f(1/2) + o(1) \textrm{ as } n \to +\infty.$$


Then we simply use the density of smooth functions in the space of continuous functions on $$[0,1]$$ to extend the result to all continuous functions on $$[0,1]$$. 
