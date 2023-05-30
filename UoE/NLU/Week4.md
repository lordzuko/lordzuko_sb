# Word Embeddings

Readings:
* [Efficient estimation of word representations in vector space](https://arxiv.org/abs/1301.3781). Mikolov et al., NIPS Workshop 2013.
-   [Contextual word representations: A contextual](https://arxiv.org/abs/1902.06006) [introduction.](https://arxiv.org/abs/1902.06006) Smith, 2019. This paper provides a conceptual overview of word embeddings, and explains why contextualized embeddings are an important innovation


* **Pre-training**: train a generic source model (e.g., VGG-16) on a standard, large dataset (e.g., ImageNet). You can think of pre-training as a way of initializing the parameters of your target model to good values.
* **Finetuning**: then take the resulting model, keep its parameters, and replace the output layer to suit the new task. Now train this target model on the dataset for the new task. Transfer learning by finetuning.

While finetuning, keep following things in mind:
* we typically have fewer output classes in the finetuning class
* use smaller learning rate
* finetune only a handful of layer, while keeping other fixed: **weight freezing**
* **source model in finetuning is usually a neural language model**, however the target task is very different from that, as the target task is rarely **Next-Word-Prediction** 
* We can all weights of embedding layer for the target model to be trained - this is a limited form of **finetuning.**  - We can do full-scale finetuning for NLP, which is possible because of contextualized word embeddings.



* **Static Word Embeddings**:
	* Eg: **Word2Vec**, **Glove**, **FastText**
	* They assign fixed vector to each word; **context independent**
	* Used for initializing the embedding layers of a target model; i.e. for feature extraction
	* **Not designed for finetuning**
	* efficient and can be trained from scratch with humble resources
*  **Contextualized Word Embeddings**
	* Eg: Bert, GPT etc.
	* Often called pretrained language model or **LLM (large language models)
	* assign a vector to a word that depends on its context: i.e. on its preceding and following words - **context dependent**
	* can be used in target model to initialize embeddings like static embeddings
	* **designed to be finetuned** - we re-train some of the weights of the embedding model for the target task.
	* **requires considerable resources to train from scratch**; finetuning is efficient way of using them


# Static Word Embeddings

* **Word2Vec : Continuous Bag-of-words model (CBOW)**
* ![[Screenshot 2023-05-02 at 11.19.53 PM.png|400]]
* **CBOW**: uses words within a context window to predict the current word (5 window size is used)
* architecture: 
	* has only a single, linear hidden layer
	* Weights for differnt position are shared

* **Word2Vec: Skipgram**
* ![[Screenshot 2023-05-02 at 11.25.27 PM.png|400]]
* **Skipgram:** uses the current word to predict the context words
* architecture:
	* flipped version of CBOW
* **Details:**
	* Objective function maximizes the probability of the corpus D (set of all word and context pairs extracted from text):
		* $$
			
				\arg\max_\theta \prod_{(w,c) \epsilon D} p(c|w;\theta)

		$$
	* Here: $p(c|w;\theta)$  is defined as:
		$$
		p(c|w;\theta) = \frac{exp(v_c . v_w)}{\sum_{c^` \epsilon C}exp(V_{c^`} . v_w)}
	   $$

	* $v_c$, $v_w$ are vector representation of $c$ and $w$ and $C$ is the set of all available contexts
	* $W$ - word vocabulary
	* $C$ - Context vocabulary
	* $d$ - embedding dim
	* each $w  \epsilon  W$  is associated with vector $v_w \epsilon \mathbb{R}^d$
	* each $c \epsilon C$  is associated with vector $v_c \epsilon \mathbb{R}^d$

	* NOTE: the formulation is impractical due to $\sum_{c^` \epsilon C}exp(V_{c^`} . v_w)$ over all context $c^`$ . This makes the softmax too expensive to compute for large vocabulary. Negative sampling helps us make the sum tractable.

* **Negative Sampling**
	* Check if you have time: [word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method](https://arxiv.org/pdf/1402.3722.pdf) - has the derivation of negative sampling objective and good information on skip-gram model.
	* The idea is to not evaluate the full output layer based on the hidden layer. 
		* Treat this as independent logistic regression classifiers (i.e. **replace softmax objective with a binary classification objective**)
			* The model is trained to distinguish
				* the positive class / **target word**
				* a few randomly sampled neurons or **randomly sampled negative words**
			* This make training speech independent of vocabulary size and this can be parallelized

* **Detailed derivation of Negative sampling Objective:**

Consider a pair $(w, c)$ of word and context. Did this pair come from the   training data? Let’s denote by $p(D = 1|w, c)$ the probability that (w, c) came   from the corpus data. Correspondingly, $p(D = 0|w, c) = 1 − p(D = 1|w, c)$ will   be the probability that $(w, c)$ did not come from the corpus data. As before,  assume there are parameters θ controlling the distribution: $p(D = 1|w, c; θ)$.

$$ 
\begin{align}
&\quad \arg\max_{\theta} \prod_{(w,c)\in D} p(D=1|w,c;\theta)\\ &\quad = \arg\max_{\theta} \log \prod_{(w,c)\in D} p(D=1|w,c;\theta)\\ &\quad = \arg\max_{\theta} \sum_{(w,c)\in D} \log p(D=1|w,c;\theta)
\end{align}
$$
  where $p(D=1|w,c;\theta)$ can be defined using softmax:
$$ \begin{align}
 
&\quad p(D=1|w,c;\theta) = \frac{1}{1+e^{-v_c \cdot v_w}} \\
\end{align}
$$
 Leading to the objective: 
$$\begin{align}
&\quad \Rightarrow \arg\max_{\theta} \sum_{(w,c)\in D} \log \frac{1}{1+e^{-v_c \cdot v_w}}
	\end{align}
$$
 This objective has a trivial solution if we set $θ$ such that $p(D = 1|w, c; θ) = 1$   for every pair $(w, c)$. This can be easily achieved by setting θ such that $v_c = v_w$  and $v_c · v_w = K$ for all $v_c$, vw , where K is large enough number (practically, we   get a probability of 1 as soon as $K ≈ 40)$.  
 We need a mechanism that prevents all the vectors from having the same  value, by disallowing some $(w, c)$ combinations. One way to do so, is to present  the model with some $(w, c)$ pairs for which $p(D = 1|w, c; θ)$ must be low, i.e.  pairs which are not in the data. This is achieved by generating the set $D^′$  of random $(w, c)$ pairs, assuming they are all incorrect (the name “negative-sampling” stems from the set $D^′$ of randomly sampled negative examples). The  optimization objective now becomes:
 
$$
\begin{align}
&\quad= \arg\max_{\theta} \prod_{(w,c)\in D} p(D=1|c,w;\theta) \prod_{(w,c)\in D'} (1 - p(D=1|c,w;\theta)) \\
 &\quad= \arg\max_{\theta} \prod_{(w,c)\in D} p(D=1|c,w;\theta) \prod_{(w,c)\in D'} (1 - p(D=1|c,w;\theta)) \\
&\quad= \arg\max_{\theta} \sum_{(w,c)\in D} \log p(D=1|c,w;\theta) + \sum_{(w,c)\in D'} \log(1-p(D=1|w,c;\theta)) \\
&\quad= \arg\max_{\theta} \sum_{(w,c)\in D} \log \frac{1}{1+e^{-v_c\cdot v_w}} + \sum_{(w,c)\in D'} \log \frac{1}{1+e^{v_c\cdot v_w}} \\
&\quad= \arg\max_{\theta} \sum_{(w,c)\in D} \log \frac{1}{1+e^{-v_c\cdot v_w}} + \sum_{(w,c)\in D'} \log \frac{1}{1+e^{v_c\cdot v_w}}
\end{align}
$$

If we let $σ(x) = \frac{1}{1+e^{−x}}$ we get:	
$$
\begin{align}
&\quad
\arg\max_{\theta} \sum_{(w,c)\in D} \log \frac{1}{1+e^{-v_c\cdot v_w}} + \sum_{(w,c)\in D'} \log \frac{1}{1+e^{v_c\cdot v_w}} \\
&\quad= \arg\max_{\theta} \sum_{(w,c)\in D} \log \sigma(v_c\cdot v_w) + \sum_{(w,c)\in D'} \log \sigma(-v_c\cdot v_w)
\end{align}
$$
This is similar to what we have in Mikolov's paper, however they assume $D^\cdot \epsilon D$  . i.e. the $(w,c)$ negative samples are taken from the dataset itself. hence their equation is 

$$
\arg\max_{\theta} \sum_{(w,c)\in D} \log p(c|w) = \sum_{(w,c)\in D} \left( \log e^{v_c\cdot v_w} - \log \sum_{c'} e^{v_{c'}\cdot v_w} \right)
$$

---


> **why does CBOW do better on syntactic tasks, given that word position isn't a factor, and skip-gram do better on semantic tasks?**

Some types of syntactic tasks may not require, long-term context information or word order information, let's take these examples:

1) "The dogs chase the cat"

2) "The big dog barks"

3) "The dog that chases the cat barks loudly"

In case 1) the relationship between the word "dogs" is in the local context of the verb "chase", and the verb can be inferred to be in the plural form to match the subject.

In case 2) the word "dog" is in the local context of the adjective "big", and the adjective can be inferred to be in the singular masculine form to match the noun.

Now, given that word order information is not present in CBOW, this grammatical information may be inferred indirectly from the distributional semantics of the words which often occur together. Maybe because some tasks such as subject-verb agreement or noun-adjective agreement, can be performed based on local context without positional information, CBOW learns to focus on the syntactic properties instead of semantic relations between words. (_Neural network has a tendency to take shortcuts_).

However, in case 3) where the agreement relationship is more complex or depends on longer-range dependencies, positional information may be necessary to capture the relationship accurately. In this case, CBOW may not perform well and skip-gram which better captures semantic relationships may serve better.

now; why does skip-gram do better on semantic tasks?

It is probably because of the architecture; skip-gram predicts the neighbouring words, which requires more semantic information. By predicting the context words based on the target word, skip-gram is able to capture broader semantic relationships between words that may not be captured by CBOW.

Hopefully, this helps.  
  
This is also a good answer on SO: [https://ai.stackexchange.com/a/18637](https://ai.stackexchange.com/a/18637) if you want to check out.

---

# Pretrained Language Models

Readings:
-   [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). Devlin et al., NAACL 2019.

* Using transformers, we can build language models that represent context very well: **contextualized embeddings**. 
* These language models can be **used for feature extraction**, but also in a **pre-training/finetuning setup**.

### [Bert ](https://aclanthology.org/N19-1423.pdf)

* Bert (Bidirectional Encoder Representation from Transformer)

* **Purpose** : 
	* **designed for pre-training** deep bidirectional representations from unlabeled text; 
	* **conditions on left and right context** in all layers; 
	* pre-trained model can be **finetuned with one additional output layer for many tasks** (e.g., NLI, QA, sentiment); 
	* for many tasks, no modifications to the Bert architecture are required;
* **Architecture**:
	* Uses bidirectional representation - i.e. **uses the left and right context both**
		* In contrast:
			* GPT - uses only left-to-right context only (unidirectional)
			* ELMO - uses shallow concatenation of independently trained left-to-right and right-to-left LSTMs
	* **Input / Output Representation**:
		* **Input sequence:** can be <Question, Answer> pair, single sentence, or any other string of tokens; 
		* **Token Types:** 
			* 30,000 token vocabulary, represented as **WordPiece embeddings** (handles OOV words); 
			* first token is always \[**CLS**\]: aggregate sentence representation for classification tasks; 
			* sentence pairs separated by \[**SEP**\] token; and by **segment embeddings**; 
				* **segment embedding**: segment embeddings are identical across all sub-words of a segment and only differ if input strings containing multiple  segments are fed into the model simultaneously (e.g. an  input containing two sentences separated by a [SEP]  token).
				* the intuition behind segment embeddings is less obvious, especially  for unsegmented (i.e. single sentence) input
					* While absent in the original Transformer, **BERT further  adds segment embeddings to the non-contextualized input representation**. In practice, this means that the input is combined with a sequence of segment IDs (a lookup of the sequence “0 0 ... 0 1 ... 1 1” ), which are found  in a lookup table of size 2 × embedding dimensionality (768 in BERTbase).
					* Actually gets applied for the NSP task. Two sentences are fed as input to BERT and the model is asked to discriminate  usbetween the true next sentence and a randomly sampled alternative sentence occurring 50% of the time. For this task, it is important for the model to be able to distinguish between the two input sentences, hence segment embeddings are used.
				* https://aclanthology.org/2022.lrec-1.152.pdf
			* token position represented by **position embeddings**.

![[Screenshot 2023-05-03 at 11.56.28 AM.png]]
* **Pretraining and Finetuning approaches**:
	* The main reason why Bert is famous - Pretraining-Finetuning (though GPT can also be used in the same way)
	
	* **Pre-training BERT**:
	
		* **Task1: Masked LM**
			* **Try to predict the masked-tokens (this is simlar to skip-gram, where we predict the surrounding words)**
			* We simply **mask some percentage of the input  tokens at random**, and then predict those masked  tokens. We refer to this procedure as a “masked  LM” (MLM), although it is often referred to as a  **Cloze task** in the literature.
			* mask 15% of the tokens in the input sequence; train the model to predict these using the context word
				* This however creates a mismatch between pre-training and fine-tuning, since the \[MASK\] token does not appear during fine-tuning.
					* do not always replace masked words with \[MASK\], instead choose 15% of token positions at random for prediction; 
						* if $i^{th}$ token is chosen, we replace the $i^{th}$ token with: 
							1. the [MASK] token 80% of the time; 
							2. a random token 10% of the time; 
							3. the unchanged $i^{th}$ token 10% of the time. 						
				* Now use $T_i$ to predict original token with cross entropy loss.
				
		* **Task2: Next Sentence Prediction (NSP)**
			* Tasks like Question-Answering (QA) and NLI are based on relationship b/w two sentences, which is not directly captured by language modeling.
			* Solution: Pre-train for a binarized NSP task; trivially this can be generated on any monolingual corpus. If we have two sentences $A$ and $B$ such that $B$ follows $A$
				* 50% of the time $B$ is the actual next setence.
				* 50% of the time $B$ is a random sentence from the corpus.
	
	* Pretraining-Finetuning can be understood as:
		![[Screenshot 2023-05-03 at 2.19.00 PM.png]]
		* (1) sentence pairs in **paraphrasing**, 
		* (2) hypothesis-premise pairs in **entailment**, 
		* (3)  question-passage pairs in **question answering**
		* (4) a degenerate $text-∅$ pair in text classification or **sequence tagging**.
	* Token Level Task v/s Full sentence task:
		* **Token Level:**
			* At the output, the token representations are fed into an output layer for token-level tasks, such as **sequence tagging** (eg: NER, POS) or **question answering**  (predicts span), 
		* **Sentence Level:**
			* the $[CLS]$ representation is fed into an output layer for classification, such as **entailment** or **sentiment analysis** (predict class label)