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
- For each $$i$$, $$X_i^R$$ and $$X_i^B$$ are Bernoulli distributions, with Bernoulli parameters $$p_R$$ and $$p_B$$ respectively. 

Next we imagine that there is some ground truth, so that $$S_N^R \to p_R$$ and $$S_N^B \to p_B$$ as $$N_R,N_B \to +\infty$$ - this is a consequence of the **Law of Large Numbers**. The **Central Limit Theorem** tells us that


$$\frac{1}{\sqrt{N_R}} \sum_{i=1}^{N_R} X_i^R \to \mathcal{N}(p_R, \sqrt{p_R(1-p_R)})$$


$$\frac{1}{\sqrt{N_B}} \sum_{i=1}^{N_B} X_i^B \to \mathcal{N}(p_B, \sqrt{p_B(1-p_B)})$$


{% highlight ruby %}
def show
  puts "Outputting a very lo-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-ong lo-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-ong line"
  @widget = Widget(params[:id])
  respond_to do |format|
    format.html # show.html.erb
    format.json { render json: @widget }
  end
end
{% endhighlight %}



The assumption that we make here is that $$p_R = p_B$$, ie the underlying distributions are the same. Another way to write this is 
$$\frac{1}{N_R}\sum_{i=1}^{N_R} X_i^R = p_R + \frac{1}{\sqrt{N_R}} \mathcal{N}(p_R, p_R(1-p_R)) + E_1$$


$$\frac{1}{N_B}\sum_{i=1}^{N_B} X_i^B = p_B + \frac{1}{\sqrt{N_B}} \mathcal{N}(p_B, p_B(1-p_B)) + E_2$$

where both $$E_1$$ and $$E_2$$ tend to $$0$$ as $$N_R$$ and $$N_B$$ tend to infinity, and the equality is understood in the distributional sense. 

Next we make the following observations

- The assumption $$H_0$$ sets $$p_B = p_R$$. 
-  We can absorb the $$\sqrt{N_R}$$ and $$\sqrt{N_B}$$ terms into the variances of the normal distributions. 
- The difference of two normally distributed random variables $$\mathcal{N}_1$$ and $$\mathcal{N}_2$$ is again a normally distributed random variable with mean $$\mu_1 - \mu_2$$ and variances $$\sigma_1^2 + \sigma_2^2$$. 


We thus obtain
\begin{equation}
\frac{\frac{1}{N_R}\sum_{i=1}^{N_R} X_i^R - \frac{1}{N_B}\sum_{i=1}^{N_B} X_i^B }{(1/\sqrt{N_R})\sqrt{ p_R(1-p_R)} +(1/\sqrt{N_B})\sqrt{ p_B(1-p_B)}} = \mathcal{N}(0,1) + E_{3},
\end{equation}
where $E_3 \to 0$ as $N_R, N_B \to +\infty$.   But wait! We don't know what $p_B$ and $p_R$ are, even if we're assuming they're equal. Well thanks to equation \eqref{LLN}, we can approximate $p_B$ and $p_R$ by their empircal
values, and this will old for large $N_R$ and $N_B$ (lots of assumptions here!). So we define the estimators
\begin{align}
\frac{1}{N_R} \sum_{i=1}^{N_R} X_i^R  = \hat p_R\\
\frac{1}{N_B} \sum_{i=1}^{N_B} X_i^B = \hat p_B.
\end{align}

We thus conclude that the following z score is sampled from a normal distribution plus some asymptotic error. 
\begin{equation}
\frac{\hat p_R - \hat p_B}{(1/\sqrt{N_R})\sqrt{ \hat p_R(1- \hat p_R)} +(1/\sqrt{N_B})\sqrt{ \hat p_B(1- \hat p_B)}}.
\end{equation}

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
$$
\int f(p) d\mathbb{P}_n(p) &= f(1/2) +  f'(1/2) \left(\mathbb{E}(p | X_n = n) - 1/2\right) + \frac{1}{2} f''(\xi) \textrm{Var}(p | X_n=x)$$
which becomes
$$= f(1/2) + o(1) \textrm{ as } n \to +\infty.$$


Then we simply use the density of smooth functions in the space of continuous functions on $$[0,1]$$ to extend the result to all continuous functions on $$[0,1]$$. 
