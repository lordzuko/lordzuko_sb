**1. b: By differentiating, show that a necessary condition to minimize P (error) is that θ satisfies  p(θ|c1)P (c1) = p(θ|c2)P (c2)**

Leibniz's rule for differentiating an integral is a method that allows us to differentiate an integral that depends on a parameter. It is named after the German mathematician Gottfried Wilhelm Leibniz.

The rule states that if we have an integral of the form:

$F(t) = \int_{a(t)}^{b(t)} f(x,t) dx$

where $a(t)$ and $b(t)$ are functions of the parameter $t$, and if the integrand $f(x,t)$ and the functions $a(t)$ and $b(t)$ are continuous, then we can differentiate $F(t)$ with respect to $t$ by first differentiating the limits of integration with respect to $t$, and then integrating the result with respect to $x$. That is:

$\frac{d}{dt} F(t) = \int_{a(t)}^{b(t)} \frac{\partial}{\partial t} f(x,t) dx + f(b(t),t)\frac{db}{dt} - f(a(t),t)\frac{da}{dt}$

Here, $\frac{\partial}{\partial t} f(x,t)$ represents the partial derivative of the integrand with respect to $t$.

To apply this rule to differentiate the expression for P(error) with respect to $\theta$, we can first write the expression as:

$P(error) = P(c1) \int_{-\infty}^{\theta} p(x|c1) dx + P(c2) \int_{\theta}^{\infty} p(x|c2) dx$

Here, $a(\theta) = -\infty$, $b(\theta) = \theta$, and $f(x,\theta) = p(x|c1)$ for the first integral, and $a(\theta) = \theta$, $b(\theta) = \infty$, and $f(x,\theta) = p(x|c2)$ for the second integral.

Using Leibniz's rule, we can differentiate P(error) with respect to $\theta$ as follows:

$\frac{d}{d\theta} P(error) = P(c1) \frac{\partial}{\partial \theta} \int_{-\infty}^{\theta} p(x|c1) dx + P(c2) \frac{\partial}{\partial \theta} \int_{\theta}^{\infty} p(x|c2) dx + P(c2) p(\theta|c2) - P(c1) p(\theta|c1)$

Evaluating the partial derivatives using the fundamental theorem of calculus, we get:

$\frac{d}{d\theta} P(error) = P(c1) p(\theta|c1) - P(c2) p(\theta|c2)$

This is the desired result, which shows that the derivative of P(error) with respect to $\theta$ is equal to the difference between the probability density functions of the two classes evaluated at $\theta$, weighted by the prior probabilities of each class.

My simple answer:

![[Screenshot 2023-04-29 at 8.13.56 PM.png]]


**1.c: Explain why this equation does not necessarily define a unique θ or even one that minimizes  P (error). Illustrate your answer with examples.**

The equation $p(\theta|c1)P(c1) = p(\theta|c2)P(c2)$ is a necessary condition for finding the value of $\theta$ that minimizes the probability of error $P(error)$ in a two-category one-dimensional problem with the decision rule "choose $c1$ if $x > \theta$, otherwise choose $c2$." However, this equation does not necessarily define a unique value of $\theta$ or even one that minimizes $P(error)$.

**We can also have multiple minimum (local)**

There are a few reasons for this:

1.  **The equation is only a necessary condition, not a sufficient condition, for finding the value of $\theta$ that minimizes $P(error)$. Even if the equation is satisfied, there may be other values of $\theta$ that give lower values of $P(error)$.**
    
2.  The equation assumes that the probability density functions $p(x|c1)$ and $p(x|c2)$ are known and can be differentiated with respect to $\theta$. In practice, these functions may be unknown or difficult to estimate accurately, which can make it challenging to find the value of $\theta$ that minimizes $P(error)$.
    
3.  The equation assumes that the prior probabilities $P(c1)$ and $P(c2)$ are known and accurately reflect the true class probabilities. In practice, these probabilities may be unknown or incorrectly specified, which can affect the value of $\theta$ that minimizes $P(error)$.
    

T**o illustrate these points, let's consider an example. Suppose we have a two-category one-dimensional problem with the following probability density functions:

$p(x|c1) = N(x; 0, 1)$

$p(x|c2) = N(x; 2, 1)$

where $N(x; \mu, \sigma^2)$ is the normal (Gaussian) probability density function with mean $\mu$ and variance $\sigma^2$.

If we assume equal prior probabilities $P(c1) = P(c2) = 0.5$, then the equation $p(\theta|c1)P(c1) = p(\theta|c2)P(c2)$ simplifies to:

$N(\theta; 0, 1) = N(\theta; 2, 1)$

This equation has two solutions: $\theta = 0$ and $\theta = 2$. However, neither of these values minimizes $P(error)$, which can be shown to be minimized at $\theta = 1$.**

This example illustrates how the necessary condition $p(\theta|c1)P(c1) = p(\theta|c2)P(c2)$ does not necessarily define a unique value of $\theta or one that minimizes $P(error)$. In this case, the equation is satisfied at two different values of $\theta$, but neither of these values gives the optimal decision rule. Other factors, such as the shape of the probability density functions and the prior probabilities, can also influence the value of $\theta$ that minimizes $P(error)$.


**2. Suppose we have two normal distributions with the same covariance but different means.**
(a) Explain what is meant by a Bayes Classifier.

A Bayes classifier is a classification system that minimises the probability of an  
incorrect classification.

(b) Derive an expression for the Bayesian Decision Boundary (BDB) in terms of the prior proba-  
bilities P (c1) and P (c2), the means μ1 and μ2 and the common covariance Σ.


3.a Show that the true error cannot decrease if we first project the distributions to a lower dimensional space and then classify them. [Hint: you can think of this geometrically first. If you want to show it mathematically it helps to think of the data lying on a 2D space.]

Geometrically, projecting the data to a lower-dimensional space corresponds to mapping the data points onto a lower-dimensional subspace of the original feature space. This can be thought of as collapsing the data points onto a lower-dimensional plane or line.

Now, suppose we have two arbitrary distributions $p(\mathbf{x}|c_1)$ and $p(\mathbf{x}|c_2)$ with known priors $P(c_1)$ and $P(c_2)$ in a $d$-dimensional feature space, and we want to classify them using a linear decision boundary. If we project the distributions onto a lower-dimensional subspace (i.e., a lower-dimensional plane or line), then the decision boundary will also be a linear boundary in the lower-dimensional space, and we can classify the data points as belonging to one of two classes based on which side of the boundary they lie.

However, projecting the data to a lower-dimensional space can never increase the distance between the means of the two distributions, since the projection essentially collapses the data onto a lower-dimensional plane or line. As a result, the linear decision boundary in the lower-dimensional space can never be further away from the means than the linear decision boundary in the original space. This means that the classification error in the lower-dimensional space can never be lower than the classification error in the original space.

Mathematically, we can show this as follows. Let $\mathbf{x}_1$ and $\mathbf{x}_2$ be two data points sampled from the two distributions, with means $\boldsymbol{\mu}_1$ and $\boldsymbol{\mu}_2$, respectively. Let $\mathbf{w}$ be a $d$-dimensional vector that defines the projection onto the lower-dimensional space, and let $\mathbf{y}_1 = \mathbf{w}^T\mathbf{x}_1$ and $\mathbf{y}_2 = \mathbf{w}^T\mathbf{x}_2$ be the corresponding projected data points. Then, the distance between the means of the projected distributions is given by:

$|\boldsymbol{\mu}_{\mathbf{y}_1} - \boldsymbol{\mu}_{\mathbf{y}_2}| = |\mathbf{w}^T\boldsymbol{\mu}_1 - \mathbf{w}^T\boldsymbol{\mu}_2| = |\mathbf{w}^T(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)|$

Since $\mathbf{w}$ is a $d$-dimensional vector, $|\mathbf{w}^T(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)|$ is the projection of the vector $(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$ onto the subspace defined by $\mathbf{w}$. By the Cauchy-Schwarz inequality, this projection is bounded by the length of $(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$, i.e.,

$|\mathbf{w}^T(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)| \leq ||\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2||$

where $||\cdot||$ denotes the Euclidean norm. Therefore, the distance between the means of the projected distributions is always less than or equal to the distance between the means of the original distributions.

Since the classification error is proportional to the distance between the means of the distributions, it follows that the true error cannot decrease if we first project the distributions to a lower-dimensional space and then classify them. In other words , the classification error in the lower-dimensional space can never be lower than the classification error in the original space. This is true for any choice of the projection vector $\mathbf{w}$ and any choice of the lower-dimensional subspace.

Therefore, if we want to minimize the classification error, we should always use the original high-dimensional feature space for classification, unless we have a good reason to believe that the relevant information in the data lies in a lower-dimensional subspace. In practice, we can use dimensionality reduction techniques, such as principal component analysis (PCA) or linear discriminant analysis (LDA), to identify the most important dimensions or subspace that capture the relevant information in the data. However, we should always keep in mind that projecting the data to a lower-dimensional space can never improve the classification accuracy beyond what is achievable in the original high-dimensional space.
