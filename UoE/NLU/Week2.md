# Feedforward Language Model

Reading:
* [A Course in Machine Learning](http://ciml.info/), chapter 4, Daumé.
* Chapter 7 of [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), 3rd edition, Jurafsky & Martin.

### Representing n-gram probabilities with neural networks

* **For neural networks:**
	* $\mathbf{x}$ : input vector
	* $\mathbf{h}$ : hidden layer
	* $\mathbf{y}$ : output vector

**Vector Rrepresentations:** 
* one-hot encoding
* probability distribution as vector 
	* Softmax: turns any vector into a probability distribution
	* Input: a vector $\mathbf{v}$ of length $|\mathbf{v}|$
	* Output: softmax($\mathbf{v}$) is a distribution $\mathbf{u}$ over $\mathbf{|v|}$ where: $$u_i = \frac{exp (v_i)}{\sum_{j=1}^{|v|} exp(v_j)}$$
* **Logistic Regression:**$$ \mathbf{y} = softmax(\mathbf{Wx + b}) $$
* $\mathbf{W}$ and $\mathbf{b}$ are parameters of the model
* Input $\mathbf{x}$ and output $\mathbf{y}$ have different dimension, which determine the dimension of $\mathbf{W}$ and $\mathbf{b}$ 
	* $\mathbf{W}$  = $|\mathbf{y}| \times |\mathbf{x}|$ 
	* $\mathbf{b}$ is bias = $|\mathbf{y}|$

### Learning and Objective Functions

**Two views**
	* **Maximize Likelihood** : Attempt to put as much as probability on the target output
	* **Minimize Error** : Minimizes the **cross-entropy** loss, which penalizes the model for the proportion of the output probability mass that it does not assign to the target output.
		* Gradient descent: good framework, however, limited in power in practice, use the following to make it more efficient
			* with momentum
			* individual learning rates for each dimension
			* adaptive learning rate
			* decoupling step length from partial derivatives


# Recurrent Networks for Language Modeling (RNN)

Readings
* Sections 4 and 5 of [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/abs/1703.01619), Neubig

So far we have seen n-gram and Feedforward networks:
* both have fixed context width, however, linguistic dependencies can be arbitrarily long. This is where RNN come in!
 
**- Why not to use a standard network for sequence tasks? There are two problems:**
  - **Inputs, outputs can be different lengths in different examples**.
    - This can be solved for normal NNs by paddings with the maximum lengths but it's not a good solution.
  - **Doesn't share features learned across different positions of text/sequence**.
    - Using a feature sharing like in CNNs can significantly reduce the number of parameters in your model. That's what we will do in RNNs.
- Recurrent neural network doesn't have either of the two mentioned problems.

* **Glossary**
	* $x_i$ : the input word transformed into one hot encoding
	* $y_i$ : the output probability distribution
	* $\mathbf{U}$ : the weight matrix of the recurrent layer
	* $\mathbf{V}$ : the weight matrix between the input layer and the hidden layer
	* $\mathbf{W}$ : the weight matrix between the hidden layer and the output layer
	* $\sigma$ : the sigmoid activation function
	* $\mathbf{h}$ : the hidden layer

* A diagram showing a RNN (this in an unfolded representation or a complete computational graph)
		* ![[Screenshot 2023-05-02 at 5.03.07 PM.png]]
		* No markov assumption is made here: $$
		\begin{align}
		&\quad P(x_{i+1}|x_1, ...,x_i) = y_i \\
		&\quad y_i = softmax(\mathbf{Wh_i + b_2}) \\
		&\quad h_i = \sigma(\mathbf{Vx_i + Uh_{i-1} + b_1}) \\
		&\quad x_i = onehot(\mathbf{x_i})
		\end{align}
	$$
	* **Training:**
		* SGD with cross entropy
		* **BPTT (backpropagation through time)** can be done through arbitrary number of steps $\tau$. This makes RNN a deep neural network with $\tau$ depth, which can be arbitrarily large
			* In practice we implement **Truncated BPTT**, i.e. we only update gradients only through a limited number of steps.
			* If we have large number of steps $\tau$, derivatives which are backpropagated becomes vanishingly small, resulting in no weight update. This problem is called **Vanishing Gradient Problem** which prevents the RNN from learning long term dependencies.


# Long Short-term Memory Networks (LSTMs)

Readings:
-   Section 6 of [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/abs/1703.01619), Graham Neubig. 
-   [Backpropagation through Time](http://ir.hit.edu.cn/~jguo/docs/notes/bptt.pdf), Jiang Guo.

Helpful Resources: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

LSTM stands for Long Short-Term Memory, which is a type of recurrent neural network (RNN) that is designed to handle the vanishing gradient problem in traditional RNNs. LSTMs can effectively capture long-term dependencies in sequential data by allowing information to flow through "memory cells" that can be selectively read from or written to at each time step.

Here are the main steps in LSTMs:

* First we have a  **Forget gate**, which  determines how much of the previous memory cell value should be retained, from cell state $C_{t-1}$
    
    $f_t = \sigma(W_f[x_t, h_{t-1}] + b_f)$
    
![[LSTM3-focus-f.png]]

- Next step, is to decide what new information we're going to store in the cell steate.  **Input gate:** determines how much of the new input should be incorporated into the memory cell. **Candidate memory cell value:** the new value to be potentially added to the memory cell
    
    $i_t = \sigma(W_i[x_t, h_{t-1}] + b_i)$
    $\tilde{C}_t = \tanh(W_c[x_t, h_{t-1}] + b_c)$

![[LSTM3-focus-i.png]]
It's now time to update the old cell state, $C_{t-1}$ into the new cell state $C_{t}$. **Memory cell update:** combines the input, forget, and candidate values to update the memory cell value
    
    $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
    ![[LSTM3-focus-C.png]]
-   Finally, we need to decide what we’re going to output. **Output gate:** determines how much of the current memory cell value should be output. **Hidden state output:** applies the output gate to the updated memory cell value to compute the hidden state output at the current time step
    
    $o_t = \sigma(W_o[x_t, h_{t-1}] + b_o)$
    $h_t = o_t * \tanh(C_t)$
    ![[LSTM3-focus-o.png]]

where $x_t$ is the input at time step $t, h_{t-1}$ is the hidden state output from the previous time step, W and b are weight and bias parameters, \sigma is the sigmoid activation function, and * denotes element-wise multiplication.

These equations allow the LSTM to selectively forget or remember information at each time step, based on the input and the current state of the memory cell. By doing so, LSTMs can effectively capture long-term dependencies in sequential data, making them well-suited for tasks such as natural language processing and speech recognition.


**How does LSTM solve the gradient problem?**

* the memory cell is linear, so its gradient doesn’t vanish;
	*     $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
	* This is linear.
	* If $f_t$ is 1 and $i_t$ is 0 then we can simply pass the activation from previous cell state $C_{t-1}$ ; this means LSTM block can retain information indefinitely.
* All the gates are trainable; the LSTM block learns when to accept input, produce output and forget information


# Gated Recurrent Unit (GRU)

GRUs are similar to LSTMs in that they are also designed to handle the vanishing gradient problem in traditional RNNs, but they have a simpler architecture with fewer parameters. Like LSTMs, GRUs use "gates" to selectively filter information flow through the network. However, they have a more streamlined approach to gating, with just two gates: an update gate and a reset gate.

Here are the main equations used in GRUs:

- **Update gate:** determines how much of the previous hidden state should be retained

  $z_t = \sigma(W_z[x_t, h_{t-1}] + b_z)$

- **Reset gate:** determines how much of the previous hidden state should be ignored

  $r_t = \sigma(W_r[x_t, h_{t-1}] + b_r)$

- **Candidate hidden state output:** a new value to be potentially added to the hidden state output

  $\tilde{h}_t = \tanh(W_h[x_t, r_t * h_{t-1}] + b_h)$

- **Hidden state output:** a linear combination of the current and candidate hidden state values, controlled by the update gate

  $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$

where $x_t$ is the input at time step $t, h_{t-1}$ is the hidden state output from the previous time step, W and b are weight and bias parameters, and \sigma is the sigmoid activation function. * denotes element-wise multiplication.

These equations allow the GRU to selectively update or ignore information at each time step, based on the input and the previous hidden state. The update gate $z_t$ controls how much of the new candidate hidden state value $\tilde{h}_t$ should be added to the previous hidden state value $h_{t-1}$, while the reset gate $r_t$ controls how much of the previous hidden state value should be ignored. 

Overall, the GRU architecture is simpler than that of the LSTM, but it can still effectively capture long-term dependencies in sequential data. By selectively updating or ignoring information at each time step, the GRU can adapt to changes in the input and capture complex patterns in the data.


### Difference b/w GRU v/s LSTM

One of the main differences is that GRUs have a simpler architecture with fewer parameters than LSTMs. **GRUs use two gates, an update gate and a reset gate, to selectively update or ignore information in the hidden state**, whereas LSTMs have **three gates, an input gate, a forget gate, and an output gate, as well as a memory cell.**

Another difference is in the way that the hidden state is updated. In GRUs, **the candidate hidden state value is computed using a single matrix multiplication of the input and the previous hidden state**, while in LSTMs, the **candidate memory cell value is computed separately from the input and the previous hidden state, using two matrix multiplications and an element-wise multiplication.**
		In GRUs, the candidate hidden state is computed as the element-wise product of the reset gate $r_t$ and the previous hidden state $h_{t-1}$, which is then concatenated with the input $x_t$ and transformed by a weight matrix W_h and an activation function tanh. In contrast, in LSTMs, the candidate memory cell $\tilde{C}_t$ is computed separately from the input $x_t$ and the previous hidden state $h_{t-1}$, using a weight matrix $W_C$ and an activation function $tanh$.
		Furthermore, the memory cell $C_t$ in LSTMs is updated using both the forget gate $f_t$ (which controls how much of the previous memory cell state to retain) and the input gate $i_t$ (which controls how much of the new candidate memory cell value to use). The hidden state $h_t$ is then computed using the output gate $o_t$ (which controls how much of the memory cell state to output) and the updated memory cell $C_t$. In GRUs, the hidden state is computed directly using the candidate hidden state and the update gate $z_t$.



The differences in architecture between GRUs and LSTMs have implications for their performance in different tasks. For example, GRUs may be more suitable for tasks that require faster training and less complex models, while LSTMs may be more suitable for tasks that require modeling more complex dependencies and handling longer sequences of data.

