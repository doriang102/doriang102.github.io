
*Under Construction*

Recall that in the last post, our aim was, given $$y \in \mathbb{R}^n$$ and our features $$X \in \mathbb{R}^{n \times d}$$ to find $$\beta \in \mathbb{R}^d$$ which minimized:

$$ \min_{\beta} \frac{1}{N} \sum_{i=1}^N(y_i - \beta \cdot \mathbf x_i)^2.$$

We will explore the idea of over fitting and model selection in this section and see the benefits and drawbacks of various methods of parameter penalizaiton. 

# Regularized Linear Regression

### Requirement 1 - Standardization of independent and dependent variables.

This is needed since we are penalizing the coefficients $$\beta$$ equally regardless of whether we use $$L^2$$ or $$L^1$$. 

Indeed, consider the example where we have a simple rule $$ y = 2x_1 + 1 + \epsilon$$ where $$\epsilon \sim \mathcal{N}(0,1)$$, but we are seeking to learn a model with $$\mathbb{x} \in \mathbb{R}^d$$ for $$d > 1$$. Clearly we can over fit this model. 

We seek to find a model:
$$ y = \beta \cdot \mathbf{x} + \beta_0 + \epsilon_i,$$
where $$ \beta_0 \in \mathbb{R}^d$$ is non-zero. More precisely we seek to minimize

$$\sum_{k=1}^n (y_k- \beta \cdot \mathbf{x_k} - \beta_0)^2 + \lambda \|\beta+\beta_0\|_{L^p}$$


If $$\{x_k\}$$ are mean-zero centered, then the first expression has mean 0 if we choose the correct intercept for $$\beta_0$$. But what if it isn't? Let's imagine that $$\{y_k\}$$ have mean $$\mu=M$$ for $$ M > > 1$$ an to fix ideas. What will be the best choice of $$\lambda$$? For each $$\lambda > 0$$ we will find that $$\beta_0 = M$$ and so we want to shrink $$\lambda \to 0$$. What effect does this have? It means that we can't penalize any of the information in $$\beta$$ and thus you will over fit.  

$$ \hat y^i = \beta_0  + \beta_1 x_1^i + \beta_2 x_2^i + \cdots + \beta_k x_k^i $$

Let's say that our actual points are:

$$y_1 = 2x_1^1 + M + 10^{-3}$$

$$y_2 = 2x_2^1 + M - 10^{-3}$$

$$y_3 = 2x_3^1 + M + 2\cdot 10^{-3}$$


But we have $$\mathbb{R}^k$ features, so we can solve:

$$2x_1^1 + M + 10^{-3} = \beta_1 x_1^1 + \beta_2 x_1^2 + \beta_0$$

$$2x_2^1 + M - 10^{-3} = \beta_1 x_2^1 + \beta_2 x_2^2 + \beta_0$$

$$ y_1 = M + 2x_1 + $$
$$\begin{bmatrix}a & b\\c & d\end{bmatrix}$$


