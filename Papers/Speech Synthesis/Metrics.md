- [[Mean Opinion Score]] (MOS)
	* Objective evaluation metric for quality measurement


- [[Frechet Audio Distance]] (FAD)
	- Objective evaluation metric for quality measurement
	* FAD score is the distance between two multivariate Gaussian distributins (i.e. the background and evaluation embeddings)
	
- [[Structural Similarity Index]] (SSIM)
- [[Log-mel Spectrogram Mean Squared Error]] (LS-MSE)
- [[Peak Signal-to-Noise Ration]] (PSNR)

* SSIM, LS-MSE, PSNR are quantitative evaluation metrics for similarity and noise measurement
* All the computations in each metric are done in the frequency domain to compare the synthesized Mel-Spectrogram with ground truth

Reference: 
* [Vocbench: A Neural Vocoder Benchmark for Speech Synthesis](https://arxiv.org/pdf/2112.03099.pdf)
