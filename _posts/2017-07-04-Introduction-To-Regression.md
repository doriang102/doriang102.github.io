
In this follow up, we will look at Linear Regression. We will go over, in detail, the assumptions made for this model with some concrete examples. After this we will discuss over fitting and the various reguarlization methods used in practice.

## Linear Regression

Most people are familiar with ordinary least squares. Given a collection of observations $$\{(y_i, \mathbf x_i)\}$$ where $$\mathbf x_i \in \mathbb{R}^k$$ are the features in our model and $$\{y_i\}$$ are the observed dependent variables which we would like to model with $$\mathbf x_i$$. We seek to find the $$\mathbf \beta \in \mathbb{R}^k$$ which minimizes

$$ \min_{\beta} \frac{1}{N} \sum_{i=1}^N(y_i - \beta \cdot \mathbf x_i)^2.$$

This seems pretty straightforward, right?
