# Bias in Embeddings and Language Models

# Summarization

Reading
* [Get To The Point: Summarization with Pointer-Generator Networks](https://aclanthology.org/P17-1099.pdf)
* [Text Summarization with Pretrained Encoders](https://aclanthology.org/D19-1387.pdf)
* [Planning with Learned Entity Prompts for Abstractive Summarization](https://watermark.silverchair.com/tacl_a_00438.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAs4wggLKBgkqhkiG9w0BBwagggK7MIICtwIBADCCArAGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQML9ZBJlXoc5EThXYTAgEQgIICgemuaxuFlkWgZQg6GOwSCTfcKUZzH3RAxrV0KI1W22jaZA1jnX_1-i6OMTPJnE01m3nN2E0Z-jCwiH0gNiti5XW-vBVBpWKPFG7UbZqkQgs4G00Sf7RY38nvFE1qPbMuZ1rgQkMOXrjfuLMPmt-_Uyc8Edbv-vQ2MadOzK-tXZlTteVkV22rVzwhN8ptYhiXv9Kxeyl-r2Oov4NEaB4AyoJ79weu95aC-92Wxl0ItCijpYER-kq2NkXLWDRRpfAaLiJYgKTGBct2VvSt4fF-a22Hr4jh0gyRGqwlsE0ADYop0Ods7kIxsyybOTa5Ohg_X7T66A0mkeA4TFAxb0hpUUPjKJK5ScoE7xB0Y3FlkiWSBz_SHAToUTUjIG-qIu3AqynQ1mPX1SvU52CHfxN8Z5HE81Kz_wpwOmf6hy8Mvg4VsF5cnyyaqDTwD4kEjBCGkzp9b3s9Ouw_SfrbczG-5F01HzzQy9fu4mynH3o2w7EXFnIvnAHxJ_hX_hqXNgwp6eRnUQffm-6AhwCuT3GV9jPkJ5j4KbBh8qVyrJ6DfUfYr-_21yqwd9DCYJP5V4-PPjsOy0ojPX_Kxbwcd6rRIHg1N2r9HrVZewLqMkm-tqpOMQ-e5W2WFqK_RE-zQcoYt94fUuWjSw-nrDWTGYkcbXFdBi0zQuFREhhL11kuH6an3vyfaru8Noy-fReEjL6lP4lvZ1_UvDdLMTZZloF9PENlGTFg59PTkc8TCLKh8hpbPvslbODrbsAva0HMA8lyIqsglvWHQ8a0ZC0Xzf8KNV9ttmQOrMyAm-W53JXRY2yCcorEEHOXaN4i5uZl4SxQ0q_W4UbPoNybbqVC66TdlCBO)
* [Conditional Generation with a Question-Answering Blueprint](https://www.researchgate.net/publication/361734006_Conditional_Generation_with_a_Question-Answering_Blueprint/fulltext/62c277363d26d6389e907d55/Conditional-Generation-with-a-Question-Answering-Blueprint.pdf?origin=publication_detail)
* [Text-Blueprint: An Interactive Platform for Plan-based Conditional Generation](https://aclanthology.org/2023.eacl-demo.13.pdf)

* From both the examples below:
	* We can say that NLG is subjective i.e. the summaries, captions etc. could be different from different person's perspective
		* eg: child migh need simpler sentences
		* person focused on diseases will say it's a covid-super spreader event etc.
![[Screenshot 2023-05-04 at 7.19.01 PM.png]]![[Screenshot 2023-05-04 at 7.19.27 PM.png]]

* A Language model predicts `next word` given predicted words so far: $p(y_t|y_1,... y_{t-1})$
* A conditional language model predicts `next word`, given the words so far, and also some other input $x$: $p(y_t|y_1,...,y_{t-1}, x)$ , where $x$ can be image, text or a database etc.
	* For MT `->` $x=$ source sentence, $y=$ target sentence
	* For Summarization `->` $x=$ input text, $y=$ summarized text


* **Modeling Approach**
![[Screenshot 2023-05-04 at 7.25.28 PM.png]]
here $J=\frac{1}{T} \sum_{t=1}^{T} J_t$  is the loss function used to optimize the model

* **Summarization: Task Definition**
	* Given input text $x$, write summary $y$ which is shorter and contains main information of $x$. 
	* Summarization can be single-document or multi-document. Typically multi-document summarization will have overlapping content i.e. same topic or many articles about same event
		* **Single-document** means we write summary y of single document $x$. 
		* **Multi-document** means we write summary y of multiple documents $x_1,..., x_n$
	* Generally multi-document summarization is difficult:
		* redundant content - but paraphrased so there might be repetition in summary
		* $x$ in the conditional probability formulation will be now very long i.e $m \times 500$, $m$ is number of document. This is more difficult to model.

* **Type of summarization**
	* Extractive Summarization:
		* Select parts (typically sentences) of the original text to form a summary
		* Generally this is easier and restrictive (no paraphrasing)
		* Less Fluent, Less Coherent
	* Abstractive Summarization:
		* Generate new text using NLG techniques
		* More difficult but more flexible (human-like)
		* Generally more fluent and coherent if done well

* **Dataset for Summarization**
	* CNN Daily Mail:
		* Pair of news articles (average 800 words) and summaries (aka story highlights), usually 3 or 4 sentences long (average 56 words)
			* highlights are written by journalists, in a compressed telegraphic manner
			* highlight need not form coherent summary - each highligh is relatively stand-alone, with little co-referencing (i.e. he, she, her, it etc.)
		* CNN 100K pairs; Daily Mail 200K pairs

* **Modeling Summarization Task**

	* **Summarization with Sequence-to-Sequence Attentional Model**
	
		![[Screenshot 2023-05-04 at 7.44.54 PM.png]]
		* This is how the modeling is described here:
			* **Token:** Word (In summarization in this model, we used most common words for vocabulary with word as token `->` no BPE/wordpiece)
			* **Encoder:** single-layer bidirectional LSTM produces a sequence of `hidden states` $h_i$.
			* **Decoder:** single-layer unidirectional LSTM receives word embedding of previous word emitted by decoder and has `decoder state` $s_t$
			* **Attention distribution:**  This tell the decoder which words are most important in the context vector and decoder knows where to focus
				$$
			\begin{align} 
				&\quad e_i^t= v^T tanh(W_h h_i+W_s s_t + b_{attn}) \\ &\quad a^t = softmax(e^t) 
			\end{align}
			$$
			* **Context Vector:** weighted sum of encoder hiddher states $h_i^* = \sum_{i} a_i^t h_i$ ; this is what decoder will see from encoder.
			* **Vocabulary Distribution:** probability distribution over all words in vocabulary, produced by decoder $P_{vocab} = \text{softmax}(V'(V([s_t, h_t^*] + b) + b'))$ ; $s_t$ is the decoder state; $V'$ is not important so we can ignore 
			* **Training Loss:** for time step $t$ is negative log likelihood of target word $w_t^*$, $loss_t=-log P(w_t^*)$

		* Now the first problem with this model is that there are words which are not present in our data; we use UNK tokens in to represent them, however, it is possible that we start using them repeatedly in our summaries.
		* **Pointer Generator Network** comes to the rescue here; now instead of repeatedly using UNK tokens, we add a copying mechanism which is useful for rare words and phrases.
			* this model allows us both `copying words by pointing`, and `generating words` from a fixed vocabulary
			* on each decoder step, we calculate $p_{gen}$, probability of `generating` next word (rather than copying it)
			* we learn $p_{gen}$ during training; this balances of copying v/s generating  $$
			P(w) = p_{gen}P_{vocab}(w) + (1 - p_{gen}) \sum_{i:w_i=w}a_i^t
		   $$
			* so during prediction of next word in decoder; we look at  $p_{gen}$ ; which acts as a gate; which tells us if we generate a word from vocabulary or if we should look into the encoder and if there is a word which is should copy it.
		* Pointer Generator fixes the problem of generating UNK; however now it does not know if it has mentioned a word before and it will keep repeating it in the summary, which mean our summary will be repeatitive now. 
		* We introduce a mechanism to check for the coverage of a word; which ensure that we do not repeat words in summary. This is called **Coverage Mechanism**
			* We add a **Coverage vector** $c^t$, which tells us what has been attended so far:
				* $$c^t = \sum_{t'=0}^{t-1} a^{t'}$$
				* Use coverage vector as extra input to attention mechanism:$$
				e_i^t = v^T tanh(W_h h_i + W_s s_t + W_c c_i^t + b_{attn})
				$$
				* **Coverage loss**: is added to training objective which penalizes overlap between coverage vector $c^t$ and new attention distribution $a^t$: $$
				covloss_t = \sum_{i} min (a_i^t, c_i^t)
				$$
					* this adds a penalty which promotes coverage if the word has not been attended to and discourages coverage from attending to already covered words
	
	*  **Summarization with Pre-Trained Encoders**
		* Pre-trained encoders like BERT are very successful in may NLU tasks, however for they were not setup for summarization. Why?
			* BERT is trained on sentence level (sentence pairs), will it work on document?
			* Here's the encoder for BERT: ![[Screenshot 2023-05-05 at 12.47.57 AM.png]]
			* How to make it ready for document as input? ![[Screenshot 2023-05-05 at 12.48.47 AM.png]]
				* as we can see we have inserted [CLS] token at the start of each sentence; also sengment embeddings tells us where a new sentence start and end
		
		* However this change brings another challenge:
			* there is a mismatch between encoder and decoder:
				* difference between how they are setup and optimized - our decoder has not seen any data whereas our encoder has seen a lot of data.
				* this implies that same training strategy won't work for both
			* How to fix this mismatch?
				* fine-tune strategy
					* learning rate schedule (vaswani et al. 2017)
						* ${lr} = \tilde{lr}.min(step^{-0.5}, step.warmup^{-1.5})$
				* we choose a `smaller learning rate` and `longer warming-up` for the encoder:
					* $\tilde{lr}_e = 2e^{-3}, warmup_e = 20,000$ 
				* we choose `larger learning rate` and `shorter warming-up` for the decoder:
					* $\tilde{lr}_e = 0.1, warmup_e = 10,000$ 

* **Evaluation of Summarization**
	* **ROUGE** : Recall-Oriented Understudy for Gisting Evaluation$$
	ROUGE-N = \frac{\sum\limits_{S\epsilon{ReferenceSummaries}}  \sum\limits_{gram_n \epsilon S}{Count_{match}(gram_n)}}{\sum\limits_{S\epsilon{ReferenceSummaries}}\sum\limits_{gram_n \epsilon S}{Count(gram_n)}}
	$$
	* Like BLEU, it is based on n-gram overlap 
	* ROUGE has no brevity penalty and is based on recall 
	* Often F1 (combination of precision and recall)  and ROUGE is reported 
	* Most commonly-reported ROUGE scores: **ROUGE-1** `unigram` overlap **ROUGE-2** `bigram` overlap, and **ROUGE-L** `longest common subsequence` overlap

	* **Summarization Results**
		![[Screenshot 2023-05-05 at 1.08.16 AM.png]]



* **Conditional Generation: Objectives**
	* Generate natural language towards a `communicative goal` 
	* which is `faithful` and can be `attributed` to is sources - `no hallucination`
	* while users Iexplicitly `control` generation outcome - `style - short, long` 

	* Long-form QA
		* some queries have long-form answer and they require `multiple` documents to answer - `hallucinations` and `attribution` can be more problematic

* changes to summarization systems:
	* Change the way entities are represented (Puduppully et al., 2019; Iso et al., 2019) 
	* The decoder skips low-confidence tokens (Tian et al., 2019)  
	* Encode documents hierarchically (Rhode et al., 2021)  
	* Adopt sparse attention mechanisms (Child et al., 2019; Beltagy et al., 2020) 
	* Introduce `planning` components (Puduppully et al., 2022; Narayan et al., 2022)

* **Planning with Entity Chains** (Narayan et al., 2022)
	* ![[Screenshot 2023-05-05 at 9.31.56 AM.png]]
	* **Big Idea:** We extract entities and then creates a summary which chain them together
		* however, the entities are context dependent - eg. `Titanic` - `the boat / the movie`
		* hence; if we miss the context we end up getting a wrong summary
		* **how is this control/planning component added?**
			* There is  **Questions under discussion (QUD)** theory of discourse structure which tells that during a discourse participants are mutually committed to resolving a partially structured **set of questions**  at a given point in time.
			* now; **discourse** has these questions/plans **implicitly** which are eventually turned into answers in a successful discourse
	
	* **Our plan is to turn the question into PLAN explicitly!!** how? lets see!
		* **Question-Answering Blueprints as Content Plans![[Screenshot 2023-05-05 at 9.42.49 AM.png]]**
			* Blueprints as intermediate (discrete) planning stage for conditional generation
			* Reduce faithfulness errors  
			* Increase controllability  
			* Are better for long-form inputs and outputs
			
			* **Blue Print Annotation**
				* Large-scale QA generation for output (summary, answer)
				* QA selection and filtering for final blueprint
				* **Steps for QA blueprint annotation:**
					1) Question-Answering Overgeneration - Identify **noun phrases** and **named entities** as answer candidates![[Screenshot 2023-05-05 at 9.51.37 AM.png]]
					2) **Generate questions for each answer candidate** using SQuAD trained Question Generation model (T5-11B) - try to generate as many questions as we want![[Screenshot 2023-05-05 at 9.53.50 AM.png]]
					3) **FIltering** - Perform **round-trip consistency check** (Alberti et al., 2019)! - After getting QA normally , we ask the same questions again to the summary and if we get the same answer, then our question is selected 
						![[Screenshot 2023-05-05 at 9.54.47 AM.png]]
					4)  **Filtering** - **Rheme-based selection** prioritizes new-information seeking questions![[Screenshot 2023-05-05 at 10.05.08 AM.png]]
					5) **Filtering** - **Coverage** prioritizes the selection of informative QA pairs by selecting non-overlapping ones.![[Screenshot 2023-05-05 at 10.06.36 AM.png]]
					* **QA Blueprint Annotation (Full picture)**![[Screenshot 2023-05-05 at 10.19.15 AM.png]]
					
			* **Blueprint Model**
				* ![[Screenshot 2023-05-05 at 1.01.33 PM.png]]
				* **End-to-end Blueprint Model**
					* Treat the blueprint as **prefix** maybe like in T5 model `->` Decoder blueprint `->` decode output 
					* Model ouputs the blueprint and summary
					* **Issue:** The output sequence is too long and the decoder runs out of memory
				* **Multitask Blueprint Model**
					* Two tasks that model does:
						1) answer plan and output sequence
						2) answer plan and questions (output and blueprint are generated separately during inference)
					* Issue:  a) The output is conditioned only on answers and not on the questions b) we run the decoder and get the output sequence, to get the blueprint we have to rerun the model.
					
				* **Iterative Blueprint Model**
					* Issue: a) No global plan (only local planning) b) This is slow as everything is planned sentence by sentence.

			* **Evaluating Model:**
				* Datasets:
					* **AQuaMuse** (Kulkarni et al., 2002, 2021): **long-form question answering**, simulates search engine, answer based on multiple retrieved documents.
					* **WikiCatSum** (Perez-Beltrachini et al., 2019): **topic-focused multi-document summarization**, generate Wikipedia abstracts.
					* **SummScreen** (Chen et al., 2022): **dialogue summarization**, generate summaries of TV episodes (e.g., CSI, The Bing Bang Theory). `->` `Hardest dataset as it has very long documents.`
				* Import notes:
					* ![[Screenshot 2023-05-05 at 12.44.54 PM.png]]
					* We found out that with blueprint methods, our summaries are much longer than earlier.
					* **Is this bad?**
						* Not necessarily as we will see below
						* First we evaluate using `ROUGE-N` and find out that all the blueprint based models (E2E, Multitask and Iterative) are similarly fluent as non-blueprint based model (LongT5):![[Screenshot 2023-05-05 at 12.47.39 PM.png]]
				* But can we only evaluate using ROUGE? isn't there anything else. Yes there are few more things we can do!
					* **QA-based** metric: (QAFactEval; Fabbri et al, 2022)
						* **Big Idea:** We have QA pairs, let's do QA on summary and see if we get the same answer; if we get the same answer we can claim that our **summaries are more grounded**
						* **Is blueprint more grounded?** (yes it is)
						* ![[Screenshot 2023-05-05 at 12.50.22 PM.png]]
					* **Textual Entailment based metric**
						* Quantify whether model summaries are **faithful** (*consistent text to the input document*) to input with textual entailment. $(t => h)$ **t** entails **h**?
							* if human reading $t$ will infer $h$ is most likely true![[Screenshot 2023-05-05 at 12.54.12 PM.png]]$$
							F(s) = \frac{1}{n}\sum\limits_{i=1}^n E(D, s_i)$$
							* E is a textual entailment model trained on public data (Honovich et al., 2022) 
							* $n$ is the number of sentences in the summary; $D$ is input document/s. 
							* $F(s)$ correlates well with human ratings $(\rho = 0.774)$.![[Screenshot 2023-05-05 at 12.56.26 PM.png]]
							* We can see that for SumScreen the difference is most discerable even though it is low.  Yes -> blueprint methods are more faithful
							