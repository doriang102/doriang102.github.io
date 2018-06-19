# Markov Decision Processes

{% highlight ruby %} 
import networkx as nx
points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]
G=nx.Graph()
G.add_edges_from(points_list)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()

 {% endhighlight %}

![](/img/mdp_graph.png?raw=true)

Our goal is to find a sequence of actions, denote by policty $\pi$ to maximize the total reward: $$\mathbb{E}[R \lvert \pi] = \int p(\tau \lvert \pi) R(\tau)$$
    
 where $$\tau$$ is the space of all *paths*, ie
 $$ \tau = (a_0,s_0,r_0,a_1,s_1,r_1,\cdots,a_t,s_t,r_t)$$
 
 
Now from the **Markov property** we have $$p(\tau \lvert \pi) = \prod_{t=1}^{T-1} p(s_{t+1}\lvert s_t,a_t) \pi (a_t \lvert s_t).$$

We will assume that $$\pi(a_t \lvert s_t)$$ is some stochastic policy, which has a form of

$$ \pi (a_t \lvert s_t) = \frac{e^{-\theta_{s_t,s_{t-1}}}}{\sum_k e^{-\theta_{s_t,s_k}}}$$


**The policy gradient trick:**

Observe that
$$
\begin{align}
\nabla_{\theta} \mathbb{E}[R \lvert \pi] &= \int \nabla_{\theta} p (\tau \lvert \pi) R(\tau) \\
&= \int \frac{\nabla_{\theta} p(\tau \lvert \pi)}{p(\tau \lvert \pi)} p(\tau \lvert \pi) R(\tau)\\
&= \mathbb{E}_p\left[ \nabla_{\theta} \log p(\tau \lvert \pi) R(\tau)\right]
\end{align}
$$

In other words, we can write this as an expectation once again, and therefore use samples from our data! But even more importantly:

$$
\begin{align}
\log p(\tau \lvert \pi) &= \log \prod_{t=1}^{T-1} p(s_{t+1}\lvert s_t,a_t) \pi (a_t \lvert s_t)\\
&=\sum_{t=1}^{T-1} \log p(s_{t+1} \lvert s_t,a_t) + \sum \log \pi(a_t \lvert s_t)
\end{align}
$$

The amazing thing here is that while our policy depends on our choice of $$\theta_{s_{t},s_{t-1}}$$, $$p(s_{t+1} \lvert s_t,a_t)$$ does not! 

## Our setting

In our case, we have 7 nodes labeled $k=0,1,\cdots,7$ and we make the softmax prior:

$$  \pi (k \lvert m ) = \frac{e^{-\theta_{k,m}}}{\sum_k e^{-\theta_{k,m}}},$$
which is the transition probability of going from node m to node k. How do we define the rewards?

**Choice of Reward:**

- We want to have a negative reward when we hit a barrier - ie. there is no edge in the graph. We set the reward in this case to be $$-1$$.
- We want to favor possible directions (even if they haven't lead to the reward yet). So we set the reward to be $1$ if the edge exists. 
- We define a reward of $$100$$ if our path reaches the goal: node 7.
