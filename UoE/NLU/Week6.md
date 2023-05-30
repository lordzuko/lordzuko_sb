
# Open Vocabulary Models

Readings
* [Neural Machine Translation of Rare Words with Subword Units](https://www.aclweb.org/anthology/P16-1162/), Sennrich et al. (2016)
- [BPE-Dropout: Simple and Effective Subword Regularization](https://aclanthology.org/2020.acl-main.170.pdf), Provilkov et al. (2020)


We are encoding rare and unknown words as sequences of subword units. This is based on  the intuition that various word classes are translatable via smaller units than words, for instance names (via character copying or transliteration), compounds (via compositional translation), and cognates and loanwords (via phonological and morphological transformations).

**Problem:**

**how do we represent text?** 

* 1-hot encoding 
	* lookup of word embedding for input 
	* probability distribution over vocabulary for output 
* large vocabularies 
	* increase network size 
	* decrease training and decoding speed 

* typical network vocabulary size: 10,000–100,000 symbols

**NLU and NLG are open-vocabulary problems** 
* many training corpora contain millions of word types 
* productive word formation processes (compounding; derivation) allow formation and understanding of unseen words 
	* names, numbers are morphologically simple, but open word classes 
* Rest of this class we are going to focus on translation

**Some common approaches which do not work:**
* **Ignoring rare words:**
	* replace out-of-vocabulary words with UNK 
	* a vocabulary of 50,000 words covers 95% of text  which gets you 95% of the way; if you only care about automatic metrics
	* However:
		* rare words are generally high information words:
			* Eg: We we miss the name below; we won't know the `subject` of the sentence
				* **source**: Mr `Gallagher` has offered a ray of hope. 
				* **reference**: Herr `Gallagher` hat einen hoffnungsstrahl ausgesandt .

**Some initial approaches, which are not good enough:**
* **Approximative Softmax**
	* compute softmax over "active" subset of vocabulary (20 subsets)
		* → smaller weight matrix, faster softmax 
	* at training time: vocabulary based on words occurring in training set partition 
	* at test time: determine likely target words based on source text (using cheap method like translation dictionary) 
	* limitations 
		* allows larger vocabulary, but still not open (completely new words don't have any representation)
		* network may not learn good representation of rare words (eg. a word seen only once in 20 sets, will not have a good representation)
* **Using back-off models**
	* replace rare words with **UNK** at training time 
	* when system produces **UNK**, align **UNK** to source word, and translate this with back-off method![[Screenshot 2023-05-04 at 10.49.50 AM.png]]
	* limitations:
		* **compounds**: hard to model 1-to-many relationships  (assumes that there is a 1-1 mapping of words in source and target languages)
		* **morphology**: hard to predict inflection with back-off dictionary (eg. in languages like turkish with complex morphology)
		* **names**: if alphabets differ, we need transliteration 
		* **alignment**: attention model unreliable (if you use alignment, to map words, the alignment from attention could be unreliable)
* **character-based translations with phrase based models** - good results for closely related languages
* **segmentation algorithm for phrase-based SMT are too conservative** - we need aggressive segmentation for open-word vocabulary (compact, without need for back-off dictionary)

**Important approaches, which work and are generally good:**
* **Subword NMT**
	* **Subword Translation:**
		* Idea: Rare words are potentially translateable through smaller units
		* Subword segmentation can also avoid the information bottleneck of a fixed-length representation - i.e. in character models when we have too long words, joining embeddings could loose information.
		* Potential category or words:
			* **named entities**: Between languages that share an alphabet, names can often be copied from source to target text. Transcription or transliteration may be required, especially if the alphabets or syllabaries differ. Example:  
				* Barack Obama (English; German)  
				* バラク・オバマ (ba-ra-ku o-ba-ma) (Japanese)
			* **cognates and loanwords**: Cognates and loanwords with a common origin can differ in regular ways between languages, so that character-level translation rules are sufficient . Example:  
				* claustrophobia (English)  
				* Klaustrophobie (German)
			* **morphologically complex words**: Words containing multiple morphemes, for instance formed via compounding, affixation, or inflection, may be translatable by translating the morphemes separately. Example:
				* solar system (English)  
				* Sonnensystem (Sonne + System) (German)  
				* Naprendszer (Nap + Rendszer) (Hungarian)
	* **Bye Pair Encoding (BPE)** (A method that actually works)
		* merge frequent pairs of characters or character sequences.
		* **why BPE**? 
			* **open-vocabulary**: operations learned on training set can be applied to unknown words 
			* **compression of frequent character sequences improves efficiency** → trade-off between text length and vocabulary size
		* **Algorithm**:
			![[Screenshot 2023-05-04 at 12.09.27 PM.png]]
		* two strategies:
			* apply BPE separately for source and target language
			* **apply BPE jointly for source and target language** (**Shared BPE**):
				* translitration is used such that both language have same characters
				* **Big Idea:** *If we apply BPE independently, the same name may be segmented differently in the two languages, which makes it harder for the neural models to learn a mapping between the subword units.*
		* Example:
			![[Screenshot 2023-05-04 at 12.06.40 PM.png|400]]
	* **BPE-Dropout**
		* Adding stochastic noise to increase model robustness 
		* **Idea:** 
			* In BPE most frequent words are intact in vocabulary, learns how to compose with infrequent words 
			* If we sometimes forget to merge, we will learn how words compose, and better transliteration 
			* forget 1 in 10 times for most scripts, 6/10 in CKJ scripts 
		* **Algorithm:**
			![[Screenshot 2023-05-04 at 12.10.17 PM.png]]
		* Consistently give 1+ BLEU scores across language pairs - widely used
		* `-` (merge performed) ; `_` (red) (merge dropped) ; `_` (green) (merge performed)
			* ![[Screenshot 2023-05-04 at 12.03.41 PM.png|600]]
* **Character level NMT**
	* Character-level Models:
		* **advantages**: 
			* (mostly) open-vocabulary 
			* no heuristic or language-specific segmentation 
			* neural network can conceivably learn from raw character sequences •
		* **drawbacks**: 
			* increasing sequence length slows training/decoding (reported x2–x8 increase in training time) 
		* open questions 
			* on which level should we represent meaning? 
			* on which level should attention operate?
			* **The disappointing answer is:** whichever gives the better downstream performance

# Low Resource MT

Readings
* [Survey of Low-Resource Machine Translation](https://arxiv.org/abs/2109.00486), Haddow et al. (2021)
* [Improving Neural Machine Translation Models with Monolinguagl Data](https://aclanthology.org/P16-1009.pdf)
* [Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets](https://arxiv.org/pdf/2103.12028.pdf)
* [Transfer Learning for Low-Resource Neural Machine Translation](https://aclanthology.org/D16-1163.pdf)
* [Trivial Transfer Learning for Low-Resource Neural Machine Translation](https://aclanthology.org/W18-6325.pdf)
* [Choosing Transfer Languages for Cross-Lingual Learning](https://arxiv.org/pdf/1905.12688.pdf)
	* https://github.com/neulab/langrank
	* LangRank is a program to solve this task of automatically selecting optimal transfer languages, treating it as a ranking problem and building models that consider the aforementioned features to perform this prediction.
* [Multilingual Denoising Pre-training for Neural Machine Translation](https://aclanthology.org/2020.tacl-1.47.pdf) - mBART
	* [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf)
* [Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges](https://arxiv.org/pdf/1907.05019.pdf)
* [Facebook AI’s WMT21 News Translation Task Submission](https://arxiv.org/pdf/2108.03265.pdf)
* [An Analysis of Massively Multilingual Neural Machine Translation for  Low-Resource Languages](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.458.pdf)

***“Low-resourced”-ness is a complex problem going beyond data availability and refects systemic problems in society.*** - Masakhane

* **Corpus Creation**
	* Web crawling:
		* Extract text from websites identified as multilingual `->` Align documents then sentences `->` Collate, deduplicate and filter
		* For new languages: ask native speakers for websites where we can collect such parallel data
	* Process of large scale extraction:
		* ![[Screenshot 2023-05-04 at 2.06.35 PM.png]]
	* How can we extract parallel data?
		* Extraction from monolingual data:
			* Large collections of monolingual data contain parallel sentences: ***Common Crawl, Internet Archive***
			* How do we detect these parallel data:
				* Map sentences into a common embedding space using eg. LASER
				* Nearest neighbours to find parallel sentences
			* Eg: Datasets: **Paracrawl, WikiMatrix, CCMatrix, Samanantar**
		* Problems with large scale extraction:
			* **Tools for low-resource languages are poor**  - such techniques may work for resource rich languages, but for resource scarce languages the tools might not be evolved and may not succeed to give the required overall outcome of sufficient quality
			* **False positives can dominate** - suppose we have very less parallel data - as we are working at petabyte of data, 1% of the false data will be too much false data.
			* The crawled resource itself may not have enough parallel data
		* Quality of crawled data
			* ![[Screenshot 2023-05-04 at 2.12.37 PM.png|600]]
		
			![[Screenshot 2023-05-04 at 2.12.44 PM.png|600]]
			 * **ParaCrawl v7.1** seems to be best

* **Using Monolingual data for MT**
	* By pairing monolingual training data with an automatic back-translation, we can treat it as additional parallel training data, and we obtain substantial improvements on the WMT 15 task `English↔German (+2.8–3.7 BLEU)`, and for the low-resourced IWSLT 14 task  `Turkish→English (+2.1–3.4 BLEU)`, obtaining new state-of-the-art results
	* ![[Screenshot 2023-05-04 at 2.37.14 PM.png]]
	* Iterated back translation for 2-3 iteration is sufficient, however this can fail if initial system is too weak.

* **Using Multilingual data for MT**
	* **Big Idea:** 
		* First train a high-resource language pair (the parent model), then transfer some of the  learned parameters to the low-resource pair (the child model) to initialize and constrain training. Using our transfer learning method we improve baseline NMT models by an average of 5.6 BLEU on four low-resource language pairs
	* **How transfer learning was done exactly:** 
		* In the French–English to Uzbek–English example, as a result of the initialization, **the English word embeddings from the parent model are copied**, *but the Uzbek words are initially mapped to random French embeddings*. **The parameters of the English  embeddings are then frozen**, while *the Uzbek embeddings’ parameters are allowed to be modified, i.e. fine-tuned, during training of the child model*.
		 ![[Screenshot 2023-05-04 at 2.53.00 PM.png|600]]
		* Parent and Child do not need to be related `->` [Trivial Transfer Learning for Low-Resource Neural Machine Translation](https://aclanthology.org/W18-6325.pdf)
		* Extensive investigation of choice of parents `->` [Choosing Transfer Languages for Cross-Lingual Learning](https://arxiv.org/pdf/1905.12688.pdf)
			* Data set size and lexical overlap important

* **Transfer learning from Many Monolingual Corpora**
	* **mBART**:
		* **Data**: 25 languages from Common Crawl `->` Then finetune parallel data separately for each task
		* **Architecture:** Encoder-Decoder 
		* **Learning Method:** Our training data covers $K$ languages: $D = {D_1, . . . , D_K }$ where each $D_i$ is a collection  of monolingual documents in language $i$. We assume access to a noising function $g$, defined below, that corrupts text, and (2) train the model to predict the original text $X$ given $g(X)$. More formally, we aim to maximize $L_θ$:  $$ L_{\theta} = \sum_{D_i \epsilon D} \sum_{X \epsilon D_i} log P(X|g(X);\theta)$$ ![[Screenshot 2023-05-04 at 3.17.58 PM.png]]
		* **Objective:** loss over full text reconstruction (not just over masked spans)
			* **Noise**: 
				* **mask spans** of text: 35% of words
				* **permute the order of sentences**
		* **Token Type:** Language token for both source and target language
		* *mBART*, *mBART50* `->` Basis of much practical work on low-resource MT
		
* **Multilingual Models**:
	* **Idea:** Handle all $N$ by $N$ translation directions with a single model (instead of $O(N^2)$
	* Usually 1-n or n-1
	* Use a small number of related langauges (As not all language pair gives good result - different linguistic properties have an effect, other possibilities as well)
	* Or go big: 103 languages [Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges](https://arxiv.org/pdf/1907.05019.pdf)
	* There is a trade-off: 
		* Transfer: benefit from addition of other languages  
		* Interferance: performance is degraded due to having to  also learn to translate other languages  
	* **Pros**: Benefits are more noticeable for the many-to-English and low-resource pairs  
	* **Cons:** High-resource pairs tend to be harmed  
	* **Cons:** Massive systems require capacity
	
* **Evaluation of Low-resrouce MT**
	* Is automatic evaluation of low-resource languages harder?  
		* Metrics are designed with high-resource langauges in mind  
		* Metrics are less reliable on poor systems  
		* Lack of good test sets and human evaluations for training metrics  
	* Human evaluation is preferable for low resource language
	* Researchers need to connect to language communities

# NLP Ethics

Readings
- [The Social Impact of Natural Language Processing](http://aclweb.org/anthology/P16-2096), Hovy and Spruit (2016)
- [Ethical and social risks of harm from Language Models](https://arxiv.org/abs/2112.04359)   Weidinger et al. (2021)
- [An Introduction to Data Ethics](https://www.scu.edu/media/ethics-center/technology-ethics/IntroToDataEthics.pdf), Vallor and Rewak. 
- [A Framework for Understanding Unintended Consequences of Machine Learning](https://arxiv.org/abs/1901.10002), Suresh and Guttag (2019)

* **NLP affects people's lives**
	* We need to ask who is affected by an NLP experiment?
	* Have they consented to the experiment or participating in it? - **Facebook's contagion experiment**
	* What derived information about you is collected which you have not consented to? they can be traced from their data

* **Who do these systems harm?**
	* Who gets admitted. 
	* Who gets hired. 
	* Who gets promoted. 
	* Who receives a loan. 
	* Who receives treatment for medical problems. 
	* Who receives the death penalty.

* **Type of Risks**
	* The NLP model accurately refects natural speech, including unjust, toxic, and oppressive tendencies present in the training data.
	
	* **Discrimination, Exclusion and Toxicity**
		* **Allocational (material) harm**: discrimination
			* eg. Models that analyse CVs for recruitment can be less likely to recommend historically discriminated groups
			* eg: Accent challenge
				* 
		* **Representative harm**: exclusionary norms eg. Q: what is a family? A: a man and a woman who get married and have children, and social stereotypes (Dr. - Man | Nurse - Woman)
			* Audacious is to boldness as [religion] is to ...
				* Muslim - Terrorism
				* Jewish - Money
		* **Ofensive Behaviour**: generate toxic language

	* **Information Hazards** (Leads to Privacy and safety harms)
		* The language models could be prompted to give the private information / safety critical information which is present in it
	* **Misinformation Harms** 
		* The LM assigning high probabilities to false, misleading, nonsensical or poor quality information.
		* **Harms involve**: people believing and acting on false information
	* **Malicious Uses**
		* From humans intentionally using the LM to cause harms
		* Types of harms:
			* Illegitimate surveillance and censorship
				* Saudi Arabia monitoring social media `->` persecutes dissidents without trial, often violently
			* Facilitating fraud and impersonation scams
			* Cyber attacks
			* Misinformation campaigns


* **While solving a problem we can ask these questions:**
	* Who are the stakeholders?  
	* What could go wrong?  
	* Who could benefit, and how?  
	* Who could be harmed, and how?
	* What can you do to mitigate possible harms?