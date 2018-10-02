## Under Construction

A common derivation that is overlooked in many disucssions on statistis is the derivation of $$\chi^2$$ and the student's t distribution.

## Chi Squared

This derivation is often glossed over in many discussions of statistical significance of categorical variables. I suspect this has to do with a disconnect between the scientists who use these methods and the mathematicians who derived them. Let's start with 

$$Z = \sum_{i=1}^k X_i^2.$$

How can we derive the pdf of $$Z$$? The first thing we need to understand is what we are trying to compute. Indeed, our goal is to find

$$p(z = \sum_{i=1}^k X_i^2)$$. 

This is simply a rewriting of the above, but an important one. It allows us to understand that we are seeking to find the probability of an entire level set - in particular the level sets of $$Z$$ which are spheres. Since $$\{X_i\}_{i=1}^k$$ are independent. we can write

$$ p\left(z = \sum_{i=1}^k X_i^2\right) = \prod_{i=1}^k p\left(X_i \big \lvert \sum_{i=1}^k X_i^2 = z\right). $$

The key thing to note here is that we are trying to find the volume of a small region in $$\mathbb{R}^k$$ where $$z = \sum_{i=1}^k x_i^2$$. In this sense, using polar coordinates, we have 

$$ dV = C_kp(z) z^{\frac{k}{2}-1}dz,$$

where $$C_kz^{\frac{k}{2}-1}$$ is the area of the k dimensional sphere. 

Since each $$X_i \sim \mathcal{N}(0,1)$$ we have, letting $$z = \sum_{i=1}^k x_i^2$$,

$$ dp\left(z = \sum_{i=1}^k X_i^2\right) = C_k\frac{1}{(\sqrt{2\pi})^k} \prod_{i=1}^k e^{-x_i^2/2} z^{\frac{k}{2}-1}. $$

Rewriting the product we have 


$$ dp\left(z = \sum_{i=1}^k X_i^2\right) = C_k\frac{1}{(\sqrt{2\pi})^k} e^{-z/2} z^{\frac{k}{2}-1}.$$


## Student's t distribution

Now that we have derived $$\chi^2$$, how can we derive the student's t distribution? Recall that this is defined as 

$$ Z = \frac{\sum_{i=1}^k X_i}{\frac{1}{\sqrt{k}}\sum_{i=1}^k X_i^2}.$$

The naive hope is that we could define a variable such as

$$ Z = \frac{Z_1}{\sqrt{Z_2}/\sqrt{k}},$$

and multiply the distributions of $$Z_1$$ and $$Z_2$$. This isn't possible however since $$Z_1$$ and $$Z_2$$ are clearly not indepenedent here. 

### Before you look at the solution, think about this

Why can't we multiply $$p(z_1)$$ and $$p(z_2))$$? Because both depend on each $$X_i$$ so we can't use independence. What would be a simpler case where we could solve this? What if we restricted to the level set of $$Z_2$$ where $$Z_2$$ is constant and then derived the probability when $$Z_1$$ varied? This would indeed be a legitmate approach. 

In probablistic terms, this is really just a rewriting of the joint distirbution $$p(Z_1,Z_2)$$ as

$$p(Z) = p\left(Z_1,Z_2 \lvert Z = \frac{Z_1}{\sqrt{Z_2}/\sqrt{k}}\right)= p\left(Z_1 \lvert Z_2,Z = \frac{Z_1}{\sqrt{Z_2}/\sqrt{k}}\right)p(Z_2) . $$

We can rewrite the condition as 

$$ Z^2 Z_2 = Z_1 k$$. 

Recall that we now from the above section that $$Z_1 \sim \mathcal{N}(0,1)$$ and $$Z_2 \sim \chi_k^2$$. 

Thus

$$p\left(Z_1 \lvert Z_2,Z = \frac{Z_1}{\sqrt{Z_2}/\sqrt{k}}\right)p(Z_2) \sim e^{-z^2 z_2/2k} e^{-z_2/2} z_2^{k/2 -1} e^{-z_2/2}$$

Rewriting this we have the right side is equal to

$$ C_k e^{-z_2(1+z^2/2k)} z^{k/2-1}. $$

We need to integrate over all such $$z_2$$ though! So 

$$\int_{-\infty}^{+\infty} C_k e^{-z_2(1+z^2/2k)} z^{k/2-1} = \tilde C_k \left(1 + z^2/k\right)^{-(k+1)/2},$$

where we've used integration by parts $$k$$ times. 

### Is this really worth it?

We know that

$$ f_k(z) = \left(1 + z^2/k\right)^{-k/2} \to e^{-z^2/2} \textrm{ as } k \to +\infty.$$

Taking the log of both sides, we have 

$$ \left| f_k(z) - e^{-z^2/2} \right| = \frac{\eta(z)^4}{k^3},$$
where $$\eta(z) \in [0,z]$$. Integrating we have

$$ \left| \Phi_k(z) - \Phi_{\mathcal{N}(0,1)}(z) \right| \leq \frac{\mathbb{E}(\lvert z \rvert ^4)}{k^3}.$$

So assuming that $$Z$$ has a finite fourth moment, we have $$O(\mathbb{E}(\lvert z \rvert^4) k^{-3})$$ error estiamtes on the cdf. If we set $$M^4$$  to be the fourth moment of $$Z$$, then we need 

$$ \frac{M_4}{k^3} << 0.05.$$

Or that 

$$ k >> \left(\frac{M_4}{0.05}\right)^{1/3}.$$

