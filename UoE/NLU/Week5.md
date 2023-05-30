
### Language models are unsupervised multitask learners:

**Moving from single task to multi-task | this time by using textual prompts.**

* Learning to perform a single task can be expressed in a  probabilistic framework as estimating a conditional distribution $p(output|input)$. Since a general system should be  able to perform many different tasks, even for the same  input, it should condition not only on the input but also  on the task to be performed. That is, it should model  $p(output|input, task)$.

	For example, a translation training example can be written as the sequence 
	* $(\text{translate to french}, \text{english text}, \text{french text})$ . 
	* In order to help it infer that  this is the desired task, we condition the language model  on a context of example pairs of the format $\text{english  sentence} = \text{french sentence}$ and then after a final prompt of $\text{english sentence =}$ . The model samples and gives translation.
	Likewise, a reading comprehension training example can  be written as 
	* $(\text{answer the question}, document,  question, answer)$

**Conclusion:** 
	When a large language model is trained on a sufficiently  large and diverse dataset it is able to perform well across  many domains and datasets. The diversity of tasks the model is able to  perform in a zero-shot setting suggests that high-capacity  models trained to maximize the likelihood of a sufficiently  varied text corpus begin to learn how to perform a surprising  amount of tasks without the need for explicit supervision



# Prompting

Readings:
* Sections 1-3 [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language](https://arxiv.org/abs/2107.13586) [Processing](https://arxiv.org/abs/2107.13586), Liu et al. (2021)
* [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf), Radford et al. (2019)
* [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)- GPT

### A short overview of change in language modeling paradigm
* At the start we used to have `fully-supervised learning` modeling. Here, we were doing task specific supervised modeling
	* `feature-engineering`: domain knowledge requirement
	* then, we moved towards `architecture-engineering`, for automatic learning of features for the supervised task.
* Then we moved towards `pretrain-finetune` paradigm. `pre-train` a LM on a general purpose dataset as that is available in abundance. Adapt the `pre-trained` LM to downstream tasks by introducing additional parameters and `finetune` using task-specific objective functions.
	* focus is shifted to `objective-engineering` i.e. the training objectives to be used at both `pretraining` and `finetuning` stage. We add a pre-training task which is similar to downstream task, this improves performance on the downstream task later.
* Now we are moving towards `pre-train, prompt, predict` paradigm. Instead of adapting pre-trained LMs to downstream tasks via `objective-engineering`, we are reformulating the downstream tasks to look more like those solved during the original LM training with the help of textual `prompt`
	* Eg:`“I felt so ___” `, and ask the LM to fill the blank with an emotion-bearing word. Or if we choose the prompt `“English: I missed the bus today. French: ”`), an LM may be able to fill in the blank with a French translation.

![[Screenshot 2023-05-03 at 7.31.59 PM.png]]
### Prompting Basics
![[Screenshot 2023-05-03 at 7.35.05 PM.png]]
* **Terminologies**:
	* `prefix prompt`: variety of prompt where the input text comes entirely before $\bf{z}$
	* `cloze prompt`: the first variety of prompt with a slot to fill in the middle of the text

 * **Prompt Addition**: $f_{prompt}(x)$ is applied on $\bf{x}$ to to generate $\mathbf{x}' = f_{prompt}(x)$
	1. Apply a template, which is a textual string that has two slots: an input slot [X] for input x and an answer slot [Z] for an intermediate generated answer text z that will later be mapped into y.
	2. Fill slot [X] with the input text $\bf{x}$.
* **Answer Search**: 
	* we search for the highest-scoring text $\bf{z}ˆ$ that maximizes the score of the LM. We first define $Z$ as a set of permissible values for $\bf{z}$.
		$$
	\hat{z} = \underset{z \epsilon Z}{search} P(f_{fill}(x', z);\theta)
		$$
	* $Z$ could take variety of input:
		* **classification**: could be a small subset of the words `{“excellent”, “good”, “OK”, “bad”, “horrible”}` or `{++, +, ~, -, --}`
		* **regression**: continuous values, constants 

* **Answer Mapping**: we would like to go from the highest-scoring answer $zˆ$ to the highest-scoring output $yˆ$. This is trivial for cases, where answer itself is the output, however for cases where multiple result could result in the same output, we need a mapping function:
	* sentiment-bearing words (e.g. “excellent”, “fabulous”, “wonderful”) to represent a single class (e.g. “++”)

* **Design Considerations for Prompting**

![[Screenshot 2023-05-03 at 7.36.37 PM.png]]

### Pre-trained Language Models

Systematic view of various pre-trained LMs:
* **main training objective**
	* auxiliary training objective
* **type of text noising**
* **Directionality**: attention mask



#### Main Training Objective

The main training objective of the pre-trained LMs plays an important role in determining its applicability to particular prompting tasks.

* **Standard Language Model (SLM)**
	* Autoregressive prediction (left to right)
		* These are particularly suitable for `prefix prompts`
	
* **Denoising Objective**:
	* Noising function: $\tilde{f} = f_{noise}(x)$
	* Task to predict: $P(x|\tilde{x})$
	* These types of reconstruction objectives are suitable for `cloze prompts`
	* Two common types of denoising objectives
		* **Corrupted Text Reconstruction (CTR)**: the processed text to its uncorrupted state by calculating *loss over only the noised parts* of the input sentence
		* **Full Text Reconstruction (FTR)**: reconstruct the text by *calculating the loss over the entirety of the input texts* whether it has been noised or not
	* **Noising Functions**
		* the specific type of corruption applied to obtain the noised text $\tilde{x}$ has an effect on the efficacy of the learning algorithm
		* **prior knowledge can be incorporated by controlling the type of noise**, e.g. *the noise could focus on entities of a sentence, which allows us to learn a pre-trained model with particularly high predictive performance for entities*
		
![[Screenshot 2023-05-03 at 8.55.29 PM.png]]

* **SLM** or **FTR** objectives are maybe more suitable for *generation tasks*
* tasks such as *classification* can be formulated using models trained with any of these objectives

* **Auxiliary Training Objective**:
	* improve models’ ability to perform certain varieties of downstream tasks.
	* **Next Sentence Prediction**: Next Sentence Prediction: do two segments appear consecutively - better sentence representations - `BERT`
	* **Discourse Relation Prediction**: predict rhetorical relations between sentences - better semantics - `ERNIE [Sun et al., 2020]`
	* **Image Region Prediction**: predict the masked regions of an image - for better visual-linguistic tasks - `VL-BERT [Su et al., 2020]`



#### Directionality (Type of attention masking)

* pre-trained LM can be different based on the directionality of the calculation of representations

* **Bidirectional:** full attention no masking 
* **Left-to-right:** diagonal attention masking 
* Mix the two strategies

![[Screenshot 2023-05-03 at 9.00.35 PM.png]]

#### Typical Pre-training Methods

Following is a representation of popular pre-training methods:
![[Screenshot 2023-05-03 at 9.02.59 PM.png]]

![[Screenshot 2023-05-03 at 9.02.42 PM.png]]
* **Left-to-Right Language Model**
	* Popular backbone for many prompting methods. Representative examples of modern pre-trained left-to-right LMs include **GPT-3** , and **GPT-Neo**
	* Generally large and difficult to train - generally not available to public, thus `pretraining and finetuning`  is generally not possible
	* Useful for **generative tasks**
* **Masked Language Models**
	* Take advantage of full context. When the focus is shifted on generating optimal representation for downstream tasks.
	* BERT is a popular example which aims to predict masked text pieces based on surrounded context
	* In prompting methods, MLMs are generally most suitable for **natural language understanding or analysis tasks** (e.g., text classification, natural language inference , and extractive question answering).
	* Suitable for `cloze prompting`. 
	* `pretraining-finetuning` is generally possible
* **Prefix and Encoder-Decoder**
	* Useful for conditional text-generation tasks such as **translation** and **summarization**
		* such tasks need a pre-trained model both capable of endcoding the text and generating the output
	* (1) using an encoder with **fully-connected mask** (full-attention, no masking) to encode the source $x$ first and then (2) decode the target $y$ **auto-regressively** (from the left to right)
	* **In Prefix-LM**: Encoder-Decoder weights are shared. So same parameters are used to encode $x$ and $y$
		* Eg: UniLM 1-2, ERNIE-M
	* **In Encoder-Decoder**: Weights are different for E & D. $x$ is encoded using encoder weight whereas, $y$ is encoded using decoder weight. 
		* Eg: T5, BART
	* These models were typically used for **text generation purposes**, however, recently they are **being used for non-generation tasks** such as QA, Information Extraction etc. 


### Prompt Engineering

* Creating a promtping function $f{prompt}(x)$
* Manual template engineering
* Automated template learning of discrete prompts: 
	* Prompt mining ”[X] middle words [Z]” 
	* Paraphrase existing prompts - select the ones with highest accuracy 
* Continuous prompts: perform prompting directly in the embedding space of the model 
	* Initialise with discrete prompt, fne tune on task 
	* Template embeddings have their own parameters that can be tuned

### Training

* Promptless fne-tuning (BERT, ELMO) 
* Tuning free prompting: zero-shot (GPT3) 
* Fix prompt tune LM (T5) 
* Additional prompt parameteres: 
	* Fix LM tune prompt
	* Tune LM and prompt (high resource)

### T5 (Text-to-Text Transfer Transformer)

Readings:
* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)

![[Screenshot 2023-05-03 at 9.29.31 PM.png]]

* Model Size: up to 11B parameters - BERT-large is 330M 
* Amount of training data: 120B words of data 
* Domain/Cleanness of training data 
* Pretraining objective 
* Finetuning recipe 

* **Conclusion**: Scaling up model size and training data really helps Really easy to use pretrained model for multiple tasks using prompts!


# Evaluating Machine Translation Systems

Readings:

* [Bleu: a method for automatic evaluation of machine translation](https://aclanthology.org/P02-1040.pdf), Papenini et al. (2002)
* [COMET: A neural framework for MT evaluation](https://aclanthology.org/2020.emnlp-main.213/), Rei et al. (2020)
- [To Ship or Not to Ship: An Extensive Evaluation of Automatic Metrics for Machine Translation,](https://aclanthology.org/2021.wmt-1.57/) Kocmi et al. (2021)

 * **Why do we need to evaluate machine translation systems?**
	 * Decide which of two (or more) systems to use. 
	 * Evaluate incremental changes to systems. 
		 * Does a new idea make it better or worse? 
		 * Does it change things in the intended way? 
	 * Decide whether a system is appropriate for a given use case. 
		 * Understanding a restaurant menu. - allergy (life/death)
		 * Understanding a news about safety of a city you are visiting. 
		 * Translating legal notices of a product you are selling. (MS - 150 languages, need to get client specifications right)
		 * Negotiating a peace treaty
	 * Different level/severity of consequences:
		 * assimilate - understand what is written / do some analysis
		 * disseminate - BBC news 

* There can be different translations from different translators. A good translation is both **adequate** and **fuent**
	* **Adequate**: Does the output convey the same meaning as the input sentence? Is part of the message lost, added, or distorted?
	* **Fluent**: Is the output good fuent English? Is is grammatically correct? Does it use appropriate words and idioms?

* Axis of quality of MT:
	* Adequacy:
		* Earlier evaluated like this:
			* `5) all meaning 4) most meaning 3) much meaning 2) little meaning 1) none`
	* Fluency:
		* Earlier evaluated like this:
			* `5) flawless english 4) good english 3) non-native english 2) dis-fluent english 1) incomprehensible`
	* Rating
	* This is subjective, inconsistent, and non-reproducible.
	* High inter-annotator disagreeability

* **Goals for Evaluation Metrics**
	* **Low cost**: reduce time and money spent on carrying out evaluation  
	* **Tunable**: automatically optimize system performance towards metric  
	* **Meaningful**: score should give intuitive interpretation of translation quality  
	* **Consistent**: repeated use of metric should give same results  
	* **Correct**: metric must rank better systems higher

* **Measuring agreement between evaluators**
	$$\kappa = \frac{\rho{(A)} - \rho{(E)}}{ 1 - \rho{(E)}}$$

	* $\rho{(A)}$ is proportion of times that evaluators agree
	* $\rho{(E)}$ is proportion of times that they would agree by chance
	* Adequacy, Fluecy are very hard to measure, low but positive agreement empirically
	* Agreement on rating is relatively higher, but still low

* **Can we evaluate automatically?**
	* Idea: Use human annotated reference to automatically compare the translations
	* Requirement: Need an automatic metric

* **Automatic metrics**
	* Idea 1:
		* **Precision**: $\frac{\text{\# of correct words}}{\text{\# of output words}}$
		* **Recall**: $\frac{\text{\# of correct words}}{\text{\# of reference words}}$
		* **F-Measure**: $\frac{2 (\text{precision x recall})}{\text{precision + recall}}$
		* Eg:
			* System 1: Victory is `the opening` game `is always important`
			* Reference: It `is always important` to win `the opening` match
			* System 2: the it opening important is match always win to
			* For System 1: P $( 5 / 8)$ R $(5 / 9)$ F1 $(0.58)$
			* For System 2: P $(9 / 9)$ R $( 9/ 9)$ F1 $(1)$
		* **We see the issue that : These do not take word order into account**

	* Idea 2:
		* Count all of the n-grams that match
		* **BLEU** score - computes precision for n-grams of size 1-4 against multiple reference.
			* Recall is not defined in this setting
			* BLEU compares system length to an *effective reference length* and penalize if too short
			* Formula: $$\text{BLEU} = BP \cdot \exp(\sum_{n=1}^{N}w_n \log p_n)$$

			* where BP is the brevity penalty,  - Add brevity penalty (for too short translations)
			* $w_n$ is the weight for n-gram precision, and 
			* $p_n$ is the n-gram precision.
			* Another formula: $$
\begin{equation*}
\text{BLEU} = \min\left(1, \frac{\text{output-length}}{\text{reference-length}}\right) \left( \prod_{i=1}^4 \text{precision}_i \right)^{\frac{1}{4}}
\end{equation*}$$
		* Typically computed over the entire corpus, not single sentences
		* To account for variability, use multiple reference translations  
			– n-grams may match in any of the references  
			– closest reference length used
		  * Used for:
			* Machine Translation, Image Captioning, Text Summarization, Speech Recognition
		* Cons:
			* Human translators score low on BLEU (possibly because of higher variability, different word choices)
			* Ignore relevance of words - names and core concepts more important than determiners and punctuation
	* Idea 3: 
		* Using trained metrics
		* **COMET**  - Crosslingual Optimized Metric for Evaluation of Translation
		* Pros:
			* It has shown good correlation with human judgments of translation quality
			* It is designed to be language-independent and can be used to evaluate translations across multiple languages.
			* It takes into account the semantic similarity between the source and target sentences, which can be more meaningful than just comparing n-grams.
			* It has been shown to be more reliable and consistent than other automatic evaluation metrics, such as BLEU.
		* Cons:
			* Scores are meaningless - absolute values are not informative
			* It requires a pre-trained cross-lingual sentence encoder, which can be computationally expensive to train.
			-   It may not always align sentences correctly at the semantic level, which could lead to incorrect similarity scores.
			-   It may not be as interpretable as other metrics, such as BLEU, which simply count n-gram matches.
