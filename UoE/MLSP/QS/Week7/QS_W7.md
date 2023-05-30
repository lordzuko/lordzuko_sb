1.b **Show that the MAP estimate can change under an invertible nonlinear transform of the parameter space while the ML estimate does not.**

##### MAP vs. ML Estimates under Nonlinear Transformations

The maximum likelihood (ML) estimate of the model parameters $\theta$ is the value that maximizes the likelihood function $p(D\mid\theta)$, where $D$ represents the observed data. Mathematically, we can write:

$$
\theta_{\text{ML}} = \operatorname*{argmax}_{\theta} p(D\mid\theta)
$$

The maximum a posteriori (MAP) estimate of the parameters $\theta$ is the value that maximizes the posterior distribution $p(\theta\mid D)$. Mathematically, we can write:

$$
\theta_{\text{MAP}} = \operatorname*{argmax}_{\theta} p(\theta\mid D) = \operatorname*{argmax}_{\theta} p(D\mid\theta)p(\theta)
$$

Suppose we have a nonlinear transform $g$ of the parameter space such that $\theta = g(\phi)$, where $\phi$ represents the transformed parameters. If $g$ is invertible, we can write $\phi = g^{-1}(\theta)$. 

The likelihood function under the transformed parameter space is given by $p(D\mid\phi)$. The ML estimate of the transformed parameters $\phi$ is the value that maximizes the likelihood function, which we can write as:

$$
\phi_{\text{ML}} = \operatorname*{argmax}_{\phi} p(D\mid\phi)
$$

However, the MAP estimate of the transformed parameters $\phi$ is given by:

$$
\phi_{\text{MAP}} = \operatorname*{argmax}_{\phi} p(\phi\mid D) = \operatorname*{argmax}_{\phi} p(D\mid\phi)p(\phi)
$$

Now, we can use the **change of variables formula** for probability densities to write:

(look for the change of variables concept below)

$$
p(D\mid\phi) = p(D\mid\theta) \left|\det\left(\frac{\partial g(\phi)}{\partial\phi}\right)\right|^{-1}
$$

where $\left|\det\left(\frac{\partial g(\phi)}{\partial\phi}\right)\right|$ is the absolute value of the determinant of the Jacobian matrix of $g$ with respect to $\phi$. 

Substituting this into the expressions for the ML and MAP estimates, we get:

$$
\phi_{\text{ML}} = g^{-1}\left(\operatorname*{argmax}_{\theta} p(D\mid\theta)\right)
$$

and

$$
\phi_{\text{MAP}} = g^{-1}\left(\operatorname*{argmax}_{\theta} p(D\mid\theta)p(g(\phi))\left|\det\left(\frac{\partial g(\phi)}{\partial\phi}\right)\right|^{-1}\right)
$$

We can see that the ML estimate of the transformed parameters $\phi$ is simply the inverse transform of the ML estimate of the original parameters $\theta$. However, the MAP estimate of the transformed parameters $\phi$ depends on both the prior distribution over the transformed parameters $p(\phi)$ and the Jacobian of the transformation. Therefore, the MAP estimate can change under an invertible nonlinear transform of the parameter space while the ML estimate does not.

**Change of Variables Formula for Probability Densities**

The change of variables formula for probability densities is a formula that allows us to transform a probability density function from one variable to another. It is used when we want to work with a probability distribution in a new set of variables.

Suppose we have a probability density function $p_Y(y)$ over the random variable $Y$, and we want to transform it to a probability density function $p_X(x)$ over the random variable $X$ using a one-to-one function $g: X \rightarrow Y$. Then, the change of variables formula for probability densities states that:

$$
p_X(x) = p_Y(g(x)) \left|\frac{dg(x)}{dx}\right|
$$

where $\left|\frac{dg(x)}{dx}\right|$ is the absolute value of the derivative of the function $g$ with respect to $x$.

Intuitively, this formula tells us that the probability density of $X$ at $x$ is equal to the probability density of $Y$ at $g(x)$ multiplied by the factor that accounts for the stretching or compression of the probability density as we change variables.

To use the formula, we first find the inverse function of $g$, which we can use to express $y$ in terms of $x$. Then, we substitute this expression for $y$ into the original probability density function $p_Y(y)$ to get $p_Y(g(x))$. Finally, we multiply this by the factor $\left|\frac{dg(x)}{dx}\right|$ to get the probability density function $p_X(x)$ over the variable $X$.

The change of variables formula is commonly used in Bayesian inference and machine learning to transform probability distributions between different parameterizations or to transform data from one feature space to another.



3. **Let x be drawn from a uniform density; calculate the maximum likelihood estimate for θ given the data: D = {0.1, 0.4, 0.2, 0.8, 0.45}.**



We know that $x$ is drawn from a uniform density with parameter $\theta$ over the interval $[0, 1]$. The probability density function of $x$ is given by:

$$
p(x | \theta) =
\begin{cases}
\frac{1}{\theta} & 0 \leq x \leq \theta \\
0 & \text{otherwise}
\end{cases}
$$

Given the data $D = \{0.1, 0.4, 0.2, 0.8, 0.45\}$, the likelihood function is:

$$
L(\theta | D) = \prod_{i=1}^n p(x_i | \theta) =
\begin{cases}
\frac{1}{\theta^n} & x_{(1)} \leq \theta \leq x_{(n)} \\
0 & \text{otherwise}
\end{cases}
$$

where $x_{(1)}$ and $x_{(n)}$ are the minimum and maximum values in the dataset, respectively.

To find the maximum likelihood estimate of $\theta$, we need to find the value of $\theta$ that maximizes the likelihood function. Since the likelihood function is a decreasing function of $\theta$ for $\theta < x_{(n)}$, and an increasing function of $\theta$ for $\theta > x_{(1)}$, the maximum likelihood estimate of $\theta$ is given by:

$$
\hat{\theta}_{ML} = x_{(n)}
$$

In this case, the maximum value in the dataset is 0.8, so the maximum likelihood estimate of $\theta$ is $\hat{\theta}_{ML} = 0.8$.

**NOTE: Why are we not taking derivatives in this case? Can we also not solve for it by taking derivative of likelihood function?**

*In this particular case, the likelihood function is not differentiable everywhere, so we cannot take its derivative to find the maximum likelihood estimate of $\theta$.*

However, we can observe that the likelihood function is a decreasing function of $\theta$ for $\theta < x_{(n)}$ and an increasing function of $\theta$ for $\theta > x_{(1)}$. This means that the maximum likelihood estimate of $\theta$ must lie in the interval $[x_{(1)}, x_{(n)}]$. We can then compare the likelihood of the two endpoints of this interval to find the maximum likelihood estimate. Since the likelihood is 0 outside of this interval, we can be sure that the maximum likelihood estimate lies within it.

In this case, the endpoint $x_{(n)}$ has the highest likelihood among all the possible values of $\theta$ within the interval, so the maximum likelihood estimate is $\hat{\theta}_{ML} = x_{(n)} = 0.8$.