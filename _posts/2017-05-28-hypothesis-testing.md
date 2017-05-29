In this first blog post, I plan on discussing the detailed mathematics behind computing p values in data science, but restricted to a single hypothesis (multiple hypothesis testing will be covered later). Almost all of the expalanations I've found skip *very* important details that highlight the issues and limitations with using p values to make conclusions about the effectiveness of treatment.

### Experimental Setup

Let's assume that we have two buttons, a red button and a blue button. We wish to construct a proper experiment to test out which button results in higher conversions (clicks/likes, etc). Let $$X_i^R$$ and $$X_i^B$$ be the outcomes
of the $$ith$$ observation for the red and blue buttons respectively, ie. $$X_i^{R,B} = 1$$ if the user clicked the button, and $$X_i^{R,B} = 0$$ otherwise.  Then we define the observations after $$N_R$$ and $$N_B$$ respective trials:

\begin{align}
S_N^R &:= \frac{1}{N_R}\sum_{i=1}^{N_R} X_i^R\\
S_N^R &:= \frac{1}{N_R}\sum_{i=1}^{N_R} X_i^R.
\end{align}

Next we imagine that there is some ground truth, so that $S_N^R \to p_R$ and $S_N^B \to p_B$ as $N_R,N_B \to +\infty$ - this is a consequence of the Law of Large Numbers. The central limit theorem tells us that

\begin{align}
\frac{1}{\sqrt{N_R}} \sum_{i=1}^{N_R} X_i^R \to \mathcal{N}(p_R, \sqrt{p_R(1-p_R)})\\
\frac{1}{\sqrt{N_B}} \sum_{i=1}^{N_B} X_i^B \to \mathcal{N}(p_B, \sqrt{p_B(1-p_B)})
\end{align}

{% highlight ruby linenos %}
def show
  puts "Outputting a very lo-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-ong lo-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-ong line"
  @widget = Widget(params[:id])
  respond_to do |format|
    format.html # show.html.erb
    format.json { render json: @widget }
  end
end
{% endhighlight %}



The assumption that we make here is that $p_R = p_B$, ie the underlying distributions are the same. Another way to write this is 
\begin{align}\label{LLN}
\frac{1}{N_R}\sum_{i=1}^{N_R} X_i^R &= p_R + \frac{1}{\sqrt{N_R}} \mathcal{N}(p_R, p_R(1-p_R)) + E_1\\
\frac{1}{N_B}\sum_{i=1}^{N_B} X_i^B &= p_B + \frac{1}{\sqrt{N_B}} \mathcal{N}(p_B, p_B(1-p_B)) + E_2,
\end{align}
where both $E_1$ and $E_2$ tend to $0$ as $N_R$ and $N_B$ tend to infinity, and the equality is understood in the distributional sense. 

Next we make the following observations
\begin{itemize}
\item The assumption $H_0$ sets $p_B = p_R$. 
\item We can absorb the $\sqrt{N_R}$ and $\sqrt{N_B}$ terms into the variances of the normal distributions. 
\item The difference of two normally distributed random variables $\mathcal{N}_1$ and $\mathcal{N}_2$ is again a normally distributed random variable with mean $\mu_1 - \mu_2$ and variances $\sigma_1^2 + \sigma_2^2$. 
\end{itemize}

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

\textbf{Question:} What is the probability of observing a value equal to or larger than the above value from a normal distribution with mean 0 and variance 1? This is what a p value is. However




Assuming we have $N_R$ and $N_B$ observations for red and blue respectively, we define the conversion rate to simply be the empirical mean of our observations, so $p_R = \hat S_N^R$ and $p_B = \hat S_N^B$. 
. Now since $S_N^R$ and $S_N^B$ define random variables, we make the hypothesis that 
\begin{equation}
H_0: \; S_N^R = S_N^B.
\end{equation}
Now let's say we've done the experiment which had $N_B$ views for the blue button and $N_R$ views of the red button. We consider the joint probability distribution $p(k,j)$ for the difference of $p_R - p_J$ which is

\begin{equation}
p(k,j) = {N_R \choose k}{N_B \choose j} p_R^k (1-p_R)^{N_R-k}p_B^j (1-p_B)^{N_B-j}
\end{equation}

Under the hypothesis $H_0$, this turns into 

\begin{equation}
p(k,j|H_0) = {N_R \choose k}{N_B \choose j} p^{k+j} (1-p)^{N_R-k-j}
\end{equation}

Thus what we want to calculuate is:
\begin{equation}
p(k \geq N_R \alpha,j \leq N_B \beta|H_0) = \sum_{k \geq N_R \alpha} \sum_{j \leq N_B \beta} {N_R \choose k}{N_B \choose j} p^{k+j} (1-p)^{N_R-k-j}
\end{equation}

This is a mess though! Note however that $Z : = p_R - p_B$ satisfies
\[ \mathbb{E}(X_n) = p_R - p_B,\]
and, by independence, 
\[ \textrm{Var}(X_n) = \frac{p_R(1-p_R)}{\sqrt{N_R}} + \frac{p_B(1-p_B)}{\sqrt{N_B}}.\]

The Central Limit Theorem states that $S_n^R$  and $S_n^B$ are both approximately normally distributed for large $n$. It's also true that the sum of two normal random variables is also a normal random variable. Therefore $S_n=S_{N_R}^R - S_{N_B}^B$
is approximately normally distributed for large $N_R$ and $N_B$. We can also multiply and divide by any constants and retain normality. With this we have
\[ \frac{S_{N_R}^R - S_{N_B}^N}{p_R(1-p_R)/\sqrt{N_R} + p_B(1-p_B)/\sqrt{N_B}} \sim \mathcal{N}(0, 1) .\]
We then compute
\[ \Phi\left(\frac{p_R - p_B} {\frac{p_R(1-p_R)}{\sqrt{N_R}} + \frac{p_B(1-p_B)}{\sqrt{N_B}}}\right),\]
where $\Phi$ is the normal distribution. 
 By the hypothesis $0$ we have $\mathbb{E}(S_n^R) = \mathbb{E}(S_n^B)$. 

Thus define $z_n = \frac{p_R - p_B} {\frac{p_R(1-p_R)}{\sqrt{N_R}} + \frac{p_B(1-p_B)}{\sqrt{N_B}}.}$ and it's clear that $\mathbb{E}(Z_n) = 0$ and $\textrm{Var}(Z_n) = 1$ for all $n$. 
We then compute
\[ \alpha := \Phi\left(\frac{p_R - p_B} {\frac{p_R(1-p_R)}{\sqrt{N_R}} + \frac{p_B(1-p_B)}{\sqrt{N_B}}}\right),\]
which gives us the probability that the observed difference was by chance. However note that we are not given any kind of explicit convergence rate when we apply the central limit theorem! Thus the best we can say with this
kind of approximation is that the probability of the observed outcome is $\alpha + o(n)$ as $n \to +\infty$ - \textbf{so really tells us nothing about the actual probability}. It only tells us that \emph{if} our result holds asymptotically, then
we can compute the p value. 
\subsection{Bayes ratios}
\section{Coin Bias}
Let's define $X_{N}$ to be the number of heads obtained after $N$ flips of a coin which has bias $p$. Then $X_N$ has a distribution given by

\[ f(X_N = k | p)  = {N \choose k} p^k (1-p)^{N-k} .\]

But we really want to know what is $ f(p | X_n = k)$, ie. we want to learn the bias from the data. Thus we use Baye's theorem:

\[ f(p | X_N = k) =\frac{ f(X_N = k | p) f(p)}{f(X_n=k)}.\]

What is $f(X_n=k)$? This is summing up all possibilities over all probabilities. In other words, we have
\[ f(X_n=k) = \int_0^1 {N \choose k} p^k (1-p)^{N-k} f(p) dp.\]

Thus we have 
\[ f(p | X_N = k) =\frac{  {N \choose k} p^k (1-p)^{N-k}f(p)}{ \int_0^1 {N \choose k} p^k (1-p)^{N-k} f(p) dp}.\]

What is $f(p)$? This is just our prior belief about what $p$ is - since we don't know anything, we set it to be the \emph{uniform distribution}, so $f(p) \equiv 1$. Thus we finally have

\[ f(p | X_N = k) = \frac{ p^k(1-p)^{N-k}}{\int_0^1 p^k (1-p)^{N-k} dp}.\]
For simplicity in the next sections, are going to write $N = n + m$ so that we have
\[ f(p | X_n = m) = \frac{ p^m(1-p)^{n}}{\int_0^1 p^m (1-p)^{n} dp}.\]

Also recall that the Beta function is defined as 

\[B(n+1,m+1) = \int_0^1 p^{n} (1-p)^m dp .\]

\subsection{Case 1: $m = 0$ as  $n \to +\infty$}

In this case we have
\[ f(p | m= 0, n) = \frac{p^n}{\int_0^1 p^n dp} = (n+1)p^n .\]

It is easy to verify that the above converges in a distributional sense to $\delta_{p=1}(p)$. 
This works the same when $m = o(n)$ as $n \to +\infty$. We leave it as an exercise for the reader to ensure the calculations work the same. 

\subsection{Case 2: $m = n$ as  $n \to +\infty$}

\[\mathbb{E} (p | X_n = n) = \frac{B(n+2,n+1)}{B(n+1,n+1)} = \frac{n+1}{2n+3} \frac{B(n+1,n+1)}{B(n+1,n+1)} = \frac{n+1}{2n+3}.\]

\begin{align}\textrm{Var}(p | X_n=n) &= \frac{1}{B(n+1,n+1)} \int_0^1 (p- \frac{n+1}{2n+3})^2 p^n (1-p)^n dp
\end{align}
\[\frac{B(n+3,n+1) - B(n+2,n+1)(n+1)/(2n+3) + (n+1)^2/(2n+3)^2 B(n+1,n+1)}{B(n+1,n+1)}.\]

Using the identity $B(m+1,n) = \frac{m}{m+n} B(m,n)$ repeatedly and using the fact that $n \to + \infty$, we have 
\begin{align} \textrm{Var}(p | X_n=n) &= \frac{1}{2}\left( 1 - \frac{1}{2}\right) + o(n) \textrm{ as } n \to +\infty.\\
\mathbb{E}(p | X_n=n) &=  \frac{1}{2} +  o(n) \textrm{ as } n \to +\infty.
\end{align}

Next, we need to check if $\{Z_n = p | X_n=n\}_{n \in \mathbb{N}}$  form an independent set of random variables. Indeed, interpreting $ \{p | X_n=n\}$ as the bias on a coin for which we have observed $n$ heads out of $2n$ trials, we have

\begin{align} \mathbb{E}(Z_m Z_n) &= \frac{1}{B(m+1,m+1)B(n+1,n+1)}\int_0^1 \int_0^1 p_1 p_2 p_1^{m}(1-p_1)^{m} p_2^n(1-p_2)^n dp_1dp_2\\
&= \mathbb{E}(Z_m) \mathbb{E}(Z_n)
 \end{align}
ala Fubini's theorem. d

Thus $\{Z_i\}_{i}$ form an indepdent set of random variables with means converging to $1/2$ and variances to $\frac{1}{2}(1-\frac{1}{2})$. 


 
Now define $Z_i = p_i |( X_n = n)$ for $i = 1, \cdots, N$ (noting that $n$ and $N$ are unrelated!), and denote
\[ d\mathbb{P}_n(p) =  \frac{ p^n(1-p)^{n}}{\int_0^1 p^n(1-p)^{n} dp}.\]

Then using (0.2) and (0.3) and the Central Limit Theorem applied to $Z_i$, we have 
\begin{align}
\int_0^1\frac{1}{N} \sum_{i=1}^N  f(p_i) d\mathbb{P}_n(p_i) &=  \frac{1}{N} \sum_{i=1}^N \int_0^1 f(p) d\mathbb{P}_n(p)\\
&= \frac{N}{N} \int_0^1 f(p) d\mathbb{P}_n(p) \to \mathcal{N}(1/2 + o(n),1/4 + o(n)).
\end{align}
Sending $n \to +\infty$ proves the result. 

