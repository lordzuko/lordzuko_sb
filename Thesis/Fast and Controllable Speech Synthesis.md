Jacob Webber

Take spectrogram -> modify the F0 value 

Hider = 
Combiner  = control parameter (F0) + hidden representation

Leakage Loss = get a value of leakage loss we take the variance of output
 => the output of the finder is passed through a softmax layer to generate a sequence of prob distbn

Adversarial Losses = 
	hider generate output , the combiner can sue to recover the original

RMSE is calculated of log2(F0), log2(F1), log2(F2)

	
	![[Screenshot 2023-05-29 at 4.18.53 PM.png]]

training the finder -> two step process - 

* first 

![[Screenshot 2023-05-29 at 4.20.31 PM.png]]

reducing information in *h* means system does not ignore *y*

quality is degraded along the way:
audio -> spectrum -> modified-spectrum -> modified-audio

adversarial disentangling


differentiable iSTFT to 
autovocoder - alternative to mel-spectrogram inversion›
