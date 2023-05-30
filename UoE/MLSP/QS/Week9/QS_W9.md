Q2. 

We can use the law of total probability to express the joint probability of the observation and the current state in terms of the previous state as:

$P(V_t, \omega(t) = j | \omega_0) = \sum_i P(V_t, \omega(t) = j, \omega(t-1) = i | \omega_0)$

Using the Markov assumption, we can simplify this to:

$P(V_t, \omega(t) = j | \omega_0) = \sum_i P(V_t | \omega(t) = j, \omega(t-1) = i, \omega_0) P(\omega(t) = j | \omega(t-1) = i, \omega_0) P(\omega(t-1) = i | \omega_0)$

We can rewrite the first term using the emission probability $b_j(v_t)$, and the second term using the transition probability $a_{i,j}$:

$P(V_t, \omega(t) = j | \omega_0) = \sum_i b_j(v_t) a_{i,j} P(\omega(t-1) = i | \omega_0)$

Substituting this into the expression for $\alpha_j(t)$, we have:

$$
\begin{align}
 &\quad \alpha_j(t) = P(V_1, \dots, V_t, \omega(t) = j | \omega_0) \\
&\quad = \sum_i P(V_t, \omega(t) = j, \omega(t-1) = i | V_1, \dots, V_{t-1}, \omega_0) P(V_1, \dots, V_{t-1}, \omega(t-1) = i | \omega_0) \\
&\quad = \sum_i P(V_t | \omega(t) = j, \omega(t-1) = i, V_1, \dots, V_{t-1}, \omega_0) P(\omega(t) = j | \omega(t-1) = i, V_1, \dots, V_{t-1}, \omega_0) P(V_1, \dots, V_{t-1}, \omega(t-1) = i | \omega_0) \\
&\quad = \sum_i P(V_t | \omega(t) = j) P(\omega(t) = j | \omega(t-1) = i) P(V_1, \dots, V_{t-1}, \omega(t-1) = i | \omega_0) \\
&\quad = \sum_i b_j(v_t) a_{i,j} \alpha_i(t-1) 
\end{align} $$

So we have recursively defined $\alpha_j(t)$ as a function of the previous time step $\alpha_i(t-1)$, the transition probability $a_{i,j}$, and the emission probability $b_j(v_t)$.
