# Neural Parsing

Readings
* [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449), Vinyals et al., NeurIPS 2015. This is the encoder-decoder-based pasing model introduced in the lecture.
* [Constituency parsing with a self-attentive encoder.](https://arxiv.org/abs/1805.01052) Kitaev and Klein, ACL 2018. This is the transformer-based parsing model introduced in the lecture.

* **The Big Idea**
	* The big idea is that we can generate any structured representation that can be linearized.
		*  Eg: syntactic parsing, where the input is a string of word and the output is a tree. It's a fairly high it's a highly structured object.
		* Eg: Trees, Graphs

* Parsing is the task of turning a sequence of words: (it helps disambiguate)
	* (1) You saw a man with a telescope. into a syntax tree:![[Screenshot 2023-05-05 at 3.31.48 PM.png|500]]
* **How can we use an encoder-decoder model for parsing?**
	* **Input**: is a string
	* **Outpu**t: is not a string yet, we need to **linearize the syntax tree**:
		* `(S (NP (Pro You ) ) (VP (V saw ) (NP (Det a ) (N man ) (PP (P with ) (Det a ) (N telescope ) ) ) ) )`
		* We can even get rid of the words (is this a valid assumption?)
			* `(S (NP Pro ) (VP V (NP Det N (PP P Det N ) ) ) )`
		* And we can make it *easier to process by annotating also the closing brackets*:
			* $(S\ (NP \ Pro)_{NP} (VP\ V(NP\ Det\ N(PP\ Det\ N)_{PP} )_{NP} )_{VP} )_S$

**What could go wrong?**
* How do we know if the generated tree is valid?**
	* turns out that RNN learns to close open brackets most of the time
* **How about cross branches in syntax tree?**
	* model learns it as well

* **Encoder-Decoder for Parsing** [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449),
	* ![[Screenshot 2023-05-05 at 4.02.33 PM.png|500]]
	* To make it work properly, we need to do following:
		* **Add an end-of-sequence symbol**, as output sequences can vary in length. 
		- **Reverse the input string**: results in small performance gain. 
		- Make the **network deeper**. Vinyals et al. (2015) use three LSTM layers for both encoder and decoder. 
		- Add **attention**. This essentially works like the encoder-decoder with attention we saw for MT (lecture 7).
		- Use pre-trained word embeddings as input (here: word2vec).
		- **Get lots of training data**. Vinyals et al. (2015) use an existing parser (the Berkeley parser) to parse a large amount of text, which they then use as training data.

* **Potential Problems**
	* How do we make sure that opening and closing brackets match? Else we won’t have a well-formed tree!
		* This is really rare (0.8–1.5% of sentences). And if it occurs, just fix the brackets in post-processing (add brackets to beginning or end of the sequence).
	* How do we associate the words in the input with the leaves of the tree in the output?
		* You could just associate each input word with a PoS in the output, in sequence order. But in practice: only the tree is evaluated. Vinyals et al. (2015) replace all PoS tags with XX.
	* The output sequence can be longer than the input sequence, isn’t this a problem?
		* encoder-decoder can match sequence length mismatch; there is no restriction on sequence length both input and generated
	* How can I make sure that the model outputs the best overall sequence, not just the best symbol at each time step?
		* Use beam search to generate the output (as in MT). However, in practice, beam size has very little impact on performance. (this may not be the optimal one)
		* How about - viterbi decoding or probabilstic CYK  (guranteed optimal)

* **Evaluation**
	* Training Corpora: (part of penn tree bank was used)
		- **Wall Street Journal (WSJ)**: treebank with 40k manually annotated sentences.
		- **BerkeleyParser corpus**: 90k sentences from WSJ and several other treebanks, and 11M sentences parsed with **Berkeley Parser**.
		- **High-confidence corpus**: 90k sentences from WSJ from several treebanks, and 11M sentences for which two parsers produce the same tree (length resampled).
		- ![[Screenshot 2023-05-05 at 9.26.03 PM.png|500]]


* **Parsing with Transformers** [Constituency Parsing with a Self-Attentive Encoder](https://arxiv.org/pdf/1805.01052.pdf)

* **Architecture:**
	* ![[Screenshot 2023-05-05 at 6.47.50 PM.png|400]]
	* **Context-Aware Word Representation**
		* sub-words are used for representation of token
		* simple concatenation of character embeddings or word prefixes/suffixes can outputperform POS information from external system
		* ![[Screenshot 2023-05-05 at 6.54.24 PM.png]]
* **Content v/s Position Attention**
	* **Attention on position**:  Different locations in the sentence can attend to each other based on their positions, 
	* **Attention on content/context**:  also based on  their contents (i.e. based on the words at or around those positions)
	* We will later factor these two components in attention and we will find an improved performance in the parsing task.


# Unsupervised Parsing

Readings
- [Unsupervised parsing via constituency tests](https://arxiv.org/abs/2010.03146). Cao et al., EMNLP 2020.

**What happens when we do not have a tree bank?** Let's find out!


* **General approach of Cao. et. al:**
	- **constituency test**: 
		- take a sentence, modify it via a transformation (e.g., replace a span with a pronoun);
		- One type of constituency test involves modifying the sentence via some transformation and then judging the result.
		-   An unsupervised parser is designed by specifying a set of transformations and using an unsupervised neural acceptability model to make grammaticality decisions.
		- If the constituents pass the linguistic tests; we have evidence that a proposed constituent is an actual constituent
	
	* **Neural grammaticality models**
		- check if the result is grammatical using a neural grammaticality model;
		- produce a tree for the sentence by aggregating the results of constituency tests over spans;
		-  Specifically, we  score the likelihood that a span is a constituent by  
			applying the constituency tests and averaging their grammaticality judgments
			* $$s_{\theta}(\text{sent}, i, j) = \dfrac{1}{\lvert C \rvert}\sum_{c \in C} g_{\theta}(c(\text{sent}, i, j))$$
				* C denotes the set of constituency tests
				* $c(sent, i, j)$ is a proposed constintuent
				* a judgment function (grammaticality model) $g : sent  → \{0, 1\}$ that gives a score. If 1 -> constituent is grammaticaly otherwise not.
		* **How we learn the grammaticality model?**
			* 

	* **Parsing Algoritm**
		- select the tree with the highest score; $$t^*(sent) = arg\max_{t \in T (len(sent))} \sum_{(i,j) \in t} s_{\theta}(sent, i, j)$$
			- $T(len(sent))$ denotes the set of binary  trees with $len(sent)$ leaves.
			- If we interpret the score $s_θ(sent, i, j)$ as estimating the probability that the span $(sent, i, j)$ is a constituent, then this **formulation corresponds to choosing the tree with the highest expected number of constituents**, i.e. minimum risk decoding
				- This allows for a low probability span in the tree if the model is confident about the rest of the tree
		- we **score each tree by summing the scores**  of its spans and choose the **highest scoring binary tree via CKY**
		
	- **Refinement:**
		- add to this refinement: alternate between improving the tree and improving the grammaticality model.
		- **How does this work?**
			- 
	
