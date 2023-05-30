# Introduction

Readings:
* [The Future of Computational Linguistics: Beyond the Alchemy](https://www.frontiersin.org/articles/10.3389/frai.2021.625341/full) Church and Lieberman (21)

* ![[Screenshot 2023-05-02 at 10.16.20 AM.png]]
	* Why has deep learning taken over NLP?
		* Universal Function Approximators
		* Representation Learning
		* Multi-task learning
		* Works with wide class of data: **strings, labels, trees, graphs, tables, images**
		* deep learning solves the difficulties of applying machine learning to NLP... *But it does not solve NLP!*
	* Problems with DL:
		* Energy - carbon cost increasing exponentially
		* Ethical practice lags technical practice - *Privacy, Bias, Justice* etc.

**Fundamental Methods of the Course**

Our primary tool will be probabilistic models parameterized by  

deep learning architectures such as:  
	• feed-forward neural networks  
	• recurrent neural networks  
	• transformers  

... applied primarily to structured prediction tasks in NLP.

The second half of the course will focus on the application of deep models to a variety of core 
**NLP tasks:**  
	• machine translation  
	• word embeddings  
	• pretrained language models  
	• syntactic parsing  
	• semantic parsing  
**And applications:**  
	• paraphrasing  
	• question answering  
	• summarization  
	• data-to-text generation


# Machine Translation

Readings:
* Background Reading: [Automating Knowledge Acquisition for Machine Translation](https://kevincrawfordknight.github.io/papers/aimag97.pdf), Knight.

**Challenges in MT:**
* Words are ambiguous (**French: Banque** / **English: Bank**, **French: rive** / **English: Bank**)
* Words have complex morphology
	* Tense
	* Singular/Plural
	* Morphology:
		* Finnish: ostoskeskuksessa  
		ostos#keskus+N+Sg+Loc:in  
		shopping#center+N+Sg+Loc:in  
		English: ‘in the shopping center’
* Word order matters (reordering can change the meaning of sentence)
* Every word counts (can't omit word during translation)
* Sentences are long sometimes and complex

**Cues which might be helpful for translation:**
* Pairs of words that occur consecutively in the target language (bigrams)
* Pairs of source and target words that occur frequentyl occur together in translations

**How to model these cues? what are the ways they can fail?**
Let's look at them!!


# Conditional Language Models (with n-grams)

Readings:
* [Word Alignment and the Expectation Maximization Algorithm,](https://www.learn.ed.ac.uk/bbcswebdav/pid-7935785-dt-content-rid-30666816_1/xid-30666816_1) Lopez.
*  [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/abs/1703.01619)

**How to derive an n-gram language model**

To define the probability $P(w) = P(w1 . . . w_{|w|})$ for a given sequence of words w, we can use an n-gram language model. An n-gram model estimates the probability of a word sequence by approximating it as the product of probabilities of individual words given their previous n-1 words, where n is a positive integer (usually 1, 2, or 3).

Specifically, we can define P(w) as the product of conditional probabilities of each word given its previous n-1 words:

$$P(w) = P(w_1, w_2, \dots, w_{|w|}) \approx \prod_{i=1}^{|w|} P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1})$$

where $P(wi | w(i-1), w(i-2), ..., w(i-n+1))$ is the conditional probability of the ith word given its previous n-1 words, and the approximation holds by the Markov assumption that the probability of a word only depends on its previous n-1 words.

**Maximizing likelihood by counting n-grams**

counting n-grams is one way to estimate the probabilities in an n-gram language model, and it aims to maximize the likelihood of the observed data. The likelihood function L estimates the probability of the observed data given the model parameters (i.e., the n-gram probabilities):

$$L(P(w_1, w_2, ..., w_{|w|})) = \prod_{i=1}^{|w|} P(w_i | w_{i-1}, w_{i-2}, ..., w_{i-n+1})$$
$$
 \hat{\theta} = \arg\max_\theta P(D | \theta)
$$

where $w_1, w_2, ..., w_n$ is the observed data. The goal of estimating the n-gram probabilities is to maximize the likelihood function L, given the observed data.

Counting n-grams is a simple and effective method to estimate the n-gram probabilities, where we count the frequency of each n-gram in the training data and normalize it by the count of its preceding (n-1)-gram. However, this method has a **limitation that it assigns zero probability to any n-gram that did not occur in the training data (i.e., zero-frequency problem), which can lead to poor performance in handling unseen data.**

To overcome this limitation, **smoothing techniques are often used to estimate the probabilities of unseen n-grams.** Smoothing methods aim to assign non-zero probabilities to unseen n-grams by borrowing probabilities from other n-grams or using prior knowledge.

**Predition using n-gram LM**

If we have a sequence of words $w_1 . . . w_k$, then we can use the language model to predict the next word $w_{k+1}$:
$$w^*_{k+1} = \operatorname{argmax}_{k+1} P(w_{k+1} | w_1, w_2, \dots, w_k)$$


### Can we model Machine Translation with n-gram?

Yes, we can: but there is a lot of machinery involved such as:
Some issue: 
* Length mismatch b/w source and target language.
* Change in word order b/w source and target languages

**Machine translations is Conditional Language Modeling:**
* We will have one-word or phrase from one language mapped to the other language
* We will need to model this alignment. Now this alignment is actually unknown and unobserved, so HMM can be a good choice of model here.

$$MT = P(|y|||x|)\prod_{i=1}^{|y|} P(z_i||x|)P(y_i|x_{z_i})$$
Here: $P(|y|||x|)$  means for all pairs of sentence lengths |y|, |x|. $P(y_i|x_i)$ is the pairs of all co-occuring words in both languages. $P(z_i||x|)$ probability of how $y_i$ aligns with $x_i$ according to some distribution.

Also: $P(z_i||x|)$ is the *transition probability* and $P(y_i|x_{z_i})$ is the *emission probability*. $z$ is the latent variable here which is unobserved. We can use **Expectation-Maximization** to estimate the MLE $\hat{\theta}$  .

Now for **decoding**: 
$$
P(y|x) = \frac{P(x|y) P(y)}{P(x)} \propto  P(y) P(x|y)
$$
where: $P(y)$ is **language model** and $P(x|y)$ is the **translation model.**

**Why care about n-grams? Aren’t they obsolete?**  

1. Many of these ideas turn up again in neural models.  

	• All machine learning maximizes some **objective function**.  
	• Neural models still use **beam search**.  
	• **Latent variables** are common in **unsupervised learning**.  
	• **Alignment** directly inspired **neural attention**.  
	• Neural models exploit same signals, though more powerful.  

2. Older models are still often useful in **low-data settings**.  
3. An extension of the model in this lecture translates  n-grams to n-grams: **phrase-based translation**. It is still  used by Google for some languages, despite move to  neural MT in 2017.  
4. Understanding the tradeoffs of working with Markov assumptions will help you appreciate the fact that neural models usually make them go away!