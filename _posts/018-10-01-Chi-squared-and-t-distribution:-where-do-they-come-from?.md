## Under Construction

A common derivation that is overlooked in many disucssions on statistis is the derivation of $$\chi^2$ and the student's t distribution.

## Chi Squared

This derivation is often glossed over in many discussions of statistical significance of categorical variables. I suspect this has to do with a disconnect between the scientists who use these methods and the mathematicians who derived them. Let's start with 

$$Z = \sum_{i=1}^k X_i^2.$$

How can we derive the pdf of $Z$? The first thing we need to understand is what we are trying to compute. Indeed, our goal is to find

$$p(z = \sum_{i=1}^k X_i^2)$$. 

This is simply a rewriting of the above, but an important one. It allows us to understand that we are seeking to find the probability of an entire level set - in particular the level sets of $$Z$$ which are spheres. Since $$\{X_i\}_{i=1}^k$$ are independent. we can write

$$ p\left(z = \sum_{i=1}^k X_i^2\right) = \prod_{i=1}^k p\left(X_i \big \lvert \sum_{i=1}^k X_i^2 = z\right). $$

Since each $$X_i \sim \mathcal{N}(0,1)$$ we have, letting $$z = \sum_{i=1}^k x_i^2$$,

$$ dp\left(z = \sum_{i=1}^k X_i^2\right) = \frac{1}{(\sqrt{2\pi})^k} \prod_{i=1}^k e^{-x_i^2/2} z^{\frac{k-1}{2}}. $$

Rewriting the product we have 
$$ dp\left(z = \sum_{i=1}^k X_i^2\right) = \frac{1}{(\sqrt{2\pi})^k} e^{-z/2} z^{\frac{k-1}{2}}.$$
