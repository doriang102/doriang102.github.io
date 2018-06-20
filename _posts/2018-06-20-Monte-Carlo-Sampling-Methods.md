# UNDER CONSTRUCTION


# Metropolis Hastings Algorithm

Let's assume we can sample from some distribution $$q(x)$$ and we wish to sample from a known distribution $$p(x)$$. 

Our goal is to construct a **Markov Chain $$Q$$** whose stationary distribution is $p(x)$. This would mean that

$$\pi(x') Q(x \lvert x')A(x \lvert x') = Q(x' \lvert x) \pi(x) A (x' \lvert x),$$

for $$\pi(x) = p(x)$$ where $$Q$$ is the transition matrix and $$A$$ is the acceptance probability. The above is known as a **detailed balance**. In other words, the flow of mass $$x \mapsto x'$$ is the same as $$x' \mapsto x$$. Our acceptance probability is therefore 

$$  \frac{A(x' \lvert x)}{A(x \lvert x')} = \frac{Q(x \lvert x') p(x')}{Q(x' \lvert x) p(x)}=: H(x'|x) .$$

We want to sample $$x'$$ from $$Q(\cdot \lvert x)$$ and accept it with probability $$A(x' \lvert x)$$. 

What remains is:

* **What Q do we choose?** If we could sample from $$p$$ directly, we could just choose $$Q = p$$ with $$A = 1$$ and we would be done. But generally we can't, so we try to find something which is "close" to the distribton p in some sense. 
* **What A do we choose?** We want the probability of acceptance to be high, but any choice of $A$ satisfying the above equation will work. Note that if $$T$$ is close to $$p$$ then $$A \sim 1$$.

A common choice of $$A$$ is 

$$A(x' \lvert x) = \min \left( 1, \frac{Q(x \lvert x') p(x')}{Q(x' \lvert x) p(x)} \right) = \min \left(1, H(x' \lvert x)\right).$$



**Proof of balance:**
Note that $$H(x'\lvert x) H(x \lvert x') = 1$$. 

* **Case 1:** $$H(x' \lvert x) = 1$$. Then $$A(x' \lvert x) = A(x \lvert x') = 1$$
* **Case 2:** $$H(x' \lvert x) > 1$$. Then $$A(x' \lvert x) = 1$$. Since $$H(x \lvert x') < 1$$ by the above, we have $$A(x \lvert x') = \frac{Q(x' \lvert x) p(x)}{Q(x \lvert x') p(x')}$$ which balances the above. 
* **Case 3:** $$H(x' \lvert x) < 1$$. Then $$A(x \lvert x') = 1$$

The algorithm is defined as follows:

### Algorithm

1) Sample $$x_t \sim q(x \lvert x_{t-1})$$. 

2) Define 

$$ \alpha = \min\left\{1,\frac{q(x_t \lvert x_{t-1}) p(x_{t-1})}{q(x_{t-1} \lvert x_t) p(x_t)} \right\}.$$

3) Sample $$u \sim \textrm{Unif}[0,1]$$. 

* If $$u < \alpha$$, then accept and return $$x_t$$. 
* Otherwise set $$x_t = x_{t-1}$$. 

4) Repeat step 1).

**Example:**

{% highlight ruby %}

def p(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-(x**2)/2)


def T(xprime,x,delta=0.1):
    if np.abs(xprime-x) < delta:
        return 1
    else:
        return 0

def A(xprime,x,delta=0.1):
    alpha = min(1, (p(xprime))/p(x))
    u = np.random.uniform(0,1)
    if u < alpha:
        return 1
    else:
        return 0

def metropolis(max_iterations=100000,delta=0.1):
    x_next = 0
    x_vals=[]
    for n in range(max_iterations):
        x_candidate = np.random.uniform(x_next-delta,x_next+delta)
        
        accept = A(x_candidate,x_next)
        if accept == 1:
            x_next = x_candidate
        else:
            x_next = x_next
        x_vals.append(x_next)
    return x_vals

x_vals = metropolis()
{% endhighlight %}

## Gibbs Sampling

Note that the MH algorithm treats $p(x)$ as a black box and does not leverage any particular structure of $$p(x)$$. Similarly, the proposal distribution we choose typically does not depend on p(x) and also does not leverage any structure of $$p(x)$$.

Gibbs sampling is a MCMC algorithm that repeatedly samples from the conditional distribution of one variable of the target distribution p, given all of the other variables. Gibbs sampling works as follows:


Imagine that we have $$K$$ variables $$\mathbf X = (x_1,x_2,\cdots,x_k)$$ and we can calculate the posterior conditional distributions exactly. As before, let's try to sample from $$P(\mathbf X \lvert D)$$ by constructing a Markov Chain. 

In this case, we will choose $$i$$ randomly from $$\{1,2,\cdots,K\}$$ and hold all variables not equal to $$i$$ fixed to their previous values. We denote this vector as $$x_{-i}$$. Then we wish to sample a new value of $$x_i$$ from $$p(x_i' \lvert x_{-1})$$.

So 

$$ \alpha = \frac{ T(x_i',x_{-i} \lvert x_i, x_{i-1}) p(x_i,x_{i-1})}{T(x_i,x_{-i} \lvert x_i', x_{i-1}) p(x_i' ,x_{i-1})}. $$

Since we can compute the posteriors directly, we set

$$ T(x_i', x_{-i} \lvert x_i, x_{i-1}) = \frac{1}{K} p(x_i' \lvert x_{i-1}). $$

Then 
$$ \alpha = \frac{ p(x_i' \lvert x_{i-1}) p(x_i \lvert x_{i-1})p(x_{i-1})}{p(x_i \lvert x_{i-1}) p(x_i' \lvert x_{i-1})p(x_i' \lvert x_{-1}} = 1,$$

so we **always accept the candidate sample in Gibbs sampling**. 


