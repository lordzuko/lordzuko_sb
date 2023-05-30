# Movie Summarization

Readings
[Movie Summarization via Sparse Graph Construction.](https://arxiv.org/abs/2012.07536) Papalampidi, Keller, Lapata, 2021.

* **Motivation**
	* Summarize full-length movies by creating shorter videos containing their most informative scenes.
* **Key Concept**
	* **Turning Points**
		* Key events in a movie  that describe its storyline![[Screenshot 2023-05-05 at 2.16.21 PM.png]]
* **Problem Formulation**
	* **Data**
		* Let $D$ denote a screenplay consisting of a sequence of scenes $D = \{s_1, s_2, \ldots, s_n\}$. We aim at selecting a smaller subset $D' = \{s_i, \ldots, s_k\}$ consisting of the most informative scenes describing the movie's storyline. 
	* **Objective**
		* Our objective is to assign a binary label $y_i$ to each scene $s_i$ denoting whether it is part of the summary.
	* **Process**
		* **Classification Problem:** 
			* For each scene $s_i \in D$ we assign a binary label $y_{it}$ denoting whether it represents turning point $t$. 
			* Specifically, we calculate probabilities $p(y_{it}|s_i, D, \theta)$ quantifying the extent to which $s_i$ acts as the $t^{th}$ TP, where $t \in [1, 5]$ (and $\theta$ are model parameters). 
	* **Inference**
		* During inference, we compose a summary by selecting $l$ consecutive scenes that lie on the peak of the posterior distribution $\arg\max_{i=1}^N p(y_{it}|s_i, D, \theta)$ for each TP.


* **Modeling Process**
	* 

![[Screenshot 2023-05-05 at 2.14.35 PM.png]]
* **What does the graphs mean?**
![[Screenshot 2023-05-05 at 2.26.06 PM.png]]
