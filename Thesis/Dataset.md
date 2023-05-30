
Potential Source
https://speechresearch.github.io/prompttts/#dataset

We need to work on creating speaking style data.


Our definition of Speaking Style
**speaking style as any perceptually distinct manner of speaking that is context-appropriate.**

narration, neutral discourse, novel characters, news reader,


**News Reading**

1. The "**1996 English Broadcast News Speech (HUB4)**" dataset contains a total of 104 hours of broadcasts from ABC, CNN, and CSPAN television networks and NPR and PRI radio networks with corresponding transcripts. The dataset is divided into a training set, development data, and evaluation data. Transcripts have been made of all recordings in this publication, manually time aligned to the phrasal level, annotated to identify boundaries between news stories, speaker turn boundaries, and gender information about the speakers​[1](https://catalog.ldc.upenn.edu/LDC97S44)
2. The "**TDT4 Multilingual Broadcast News Speech Corpus**" was developed by the Linguistic Data Consortium (LDC) with support from the DARPA TIDES Program. This release contains the complete set of American English, Modern Standard Arabic, and Mandarin Chinese broadcast news audio used in the 2002 and 2003 Topic Detection and Tracking (TDT) technology evaluations, totaling approximately 607 hours. The TDT4 corpus contains news data collected daily from 20 news sources in three languages (American English, Mandarin Chinese, and Modern Standard Arabic), over a period of four months (October 2000 through January 2001)​[2](https://catalog.ldc.upenn.edu/LDC2005S11)​.​

* **BBC Programs** (Not News)
	**Lip Reading Sentences 2 (LRS2)**: This is a dataset for lip reading sentences. It's one of the largest publicly available datasets for lip reading sentences in-the-wild. The database consists mainly of news and talk shows from BBC programs. Each sentence is up to 100 characters in length. It contains thousands of speakers without speaker labels and large variation in head pose. The pre-training set contains 96,318 utterances, the training set contains 45,839 utterances, the validation set contains 1,082 utterances, and the test set contains 1,242 utterances​[1](https://paperswithcode.com/datasets?task=speech-recognition)​.
	* https://www.robots.ox.ac.uk/~vgg/data/lip_reading/



1. **LibriSpeech**: This is a standard large-scale dataset for evaluating ASR (Automatic Speech Recognition) systems. It consists of approximately 1,000 hours of narrated audiobooks collected from the LibriVox project. The speaking style is described as "narrated"​[1](https://huggingface.co/blog/audio-datasets)​.
    
2. **Common Voice**: This is a crowd-sourced open-licensed speech dataset where speakers record text from Wikipedia in various languages. The speaking style is "narrated" and the dataset contains significant variation in both audio quality and speakers. The English subset of version 11.0 contains approximately 2,300 hours of validated data​[1](https://huggingface.co/blog/audio-datasets)​.
    
3. **VoxPopuli**: This is a large-scale multilingual speech corpus consisting of data sourced from 2009-2020 European Parliament event recordings. The speaking style is described as "oratory, political speech" and the dataset largely consists of non-native speakers. The English subset contains approximately 550 hours of labeled speech​[1](https://huggingface.co/blog/audio-datasets)​.
    
4. **TED-LIUM**: This dataset is based on English-language TED Talk conference videos. The speaking style is "oratory educational talks" covering a range of different cultural, political, and academic topics. The Release 3 (latest) edition of the dataset contains approximately 450 hours of training data​[1](https://huggingface.co/blog/audio-datasets)​.
    
5. **GigaSpeech**: This is a multi-domain English speech recognition corpus curated from audiobooks, podcasts, and YouTube. It covers both "narrated and spontaneous speech" over a variety of topics. The dataset contains training splits varying from 10 hours - 10,000 hours​[1](https://huggingface.co/blog/audio-datasets)​.
    
6. **SPGISpeech**: This is an English speech recognition corpus composed of company earnings calls that have been manually transcribed. The speaking style is described as "oratory and spontaneous speech" and it contains training splits ranging from 200 hours - 5,000 hours​[1](https://huggingface.co/blog/audio-datasets)​.
    
7. **Earnings-22**: This is a 119-hour corpus of English-language earnings calls collected from global companies. The dataset does not explicitly mention the speaking style, but it is described as covering a range of real-world financial topics​[1](https://huggingface.co/blog/audio-datasets)​.