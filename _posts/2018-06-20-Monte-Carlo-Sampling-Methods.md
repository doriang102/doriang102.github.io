# UNDER CONSTRUCTION

In this approach we use the acceptance/rejection algorithm which is defined as follows. 

Let's assume we can sample from some distribution $q(x)$ and we wish to sample from a known distribution $$p(x)$$. 

Our goal is to construct a **Markov Chain $Q$** whose stationary distribution is $p(x)$. This would mean that

$$\pi(x') Q(x \lvert x')A(x \lvert x') = Q(x' \lvert x) \pi(x) A (x' \lvert x),$$

for $\pi(x) = p(x)$ where $Q$ is the transition matrix and $A$ is the acceptance probability. The above is known as a **detailed balance**. In other words, the flow of mass $x \mapsto x'$ is the same as $x' \mapsto x$. Our acceptance probability is therefore 

$$  \frac{A(x' \lvert x)}{A(x \lvert x')} = \frac{Q(x \lvert x') p(x')}{Q(x' \lvert x) p(x)}=: H(x'|x) .$$

We want to sample $x'$ from $Q(\cdot \lvert x)$ and accept it with probability $$A(x' \lvert x)$$. 

What remains is:

* ** What Q do we choose? ** If we could sample from $p$ directly, we could just choose $Q = p$ with $A = 1$ and we would be done. But generally we can't, so we try to find something which is "close" to the distribton p in some sense. 
* ** What A do we choose?** We want the probability of acceptance to be high, but any choice of $A$ satisfying the above equation will work. Note that if $T$ is close to $p$ then $A \sim 1$.

A common choice of $A$ is 

$$A(x' \lvert x) = \min \left( 1, \frac{Q(x \lvert x') p(x')}{Q(x' \lvert x) p(x)} \right) = \min \left(1, H(x' \lvert x)\right).$$



** Proof of balance: **
Note that $H(x'\lvert x) H(x \lvert x') = 1$. 

* **Case 1:** $$H(x' \lvert x) = 1$$. Then $$A(x' \lvert x) = A(x \lvert x') = 1$$
* **Case 2:** $$H(x' \lvert x) > 1$$. Then $$A(x' \lvert x) = 1$$. Since $$H(x \lvert x') < 1$$ by the above, we have $$A(x \lvert x') = \frac{Q(x' \lvert x) p(x)}{Q(x \lvert x') p(x')}$$ which balances the above. 
* **Case 3:** $$H(x' \lvert x) < 1$$. Then $$A(x \lvert x') = 1$$

The algorithm is defined as follows:

### Algorithm

1) Sample $$x_t \sim q(x \lvert x_{t-1})$$. 

2) Define 

$$ \alpha = \min\left\{1,\frac{q(x_t \lvert x_{t-1}) p(x_{t-1})}{q(x_{t-1} \lvert x_t) p(x_t)} \right\}.$$

3) Sample $u \sim \textrm{Unif}[0,1]$. 

* If $$u < \alpha$$, then accept and return $$x_t$$. 
* Otherwise set $x_t = x_{t-1}$. 

4) Repeat step 1).
