
**Topic:** **[Aligning latent space of speaking style with human perception using a re-embedding strategy [sigurgeirsson:speaking-style]](https://www.wiki.ed.ac.uk/pages/viewpage.action?spaceKey=CSTR&title=SLP+project+list+2022-23)**

To generate appropriate speech in context, we must not only consider what is said but also how it is said. Prosody is used to influence a particular meaning, to indicate attitude and emotion, and more. Modelling prosody in Text-To-Speech (TTS) is a highly complex task, as prosody manifests itself in multiple hierarchies in speech, from phone-level acoustic features to suprasegmental effects spanning multiple sentences. As perception of certain prosodic features is highly subjective and prosodically labelled speech corpora are scarce, researchers have looked to unsupervised methods of modelling prosody in TTS.

Many propose to jointly train a reference encoder to model a latent representation of prosody or speaking style [e.g., 1, 2, 3]. The reference encoder is conditioned on the target speech during training and learns to model features that are perceptually important for the acoustic model in reconstructing the target speech [1, 2]. Many recent unsupervised style-modelling methods have focused on disentangling non-prosodic features from the latent prosodic representations [e.g., 3, 4, 8], but there is little known about how well this latent space aligns with human perception of style. Understanding this space could help solve the issue of feature-entanglement without any modifications to the model.

Here, we propose to manipulate the latent prosody space using light-supervision from human annotators. Such strategies have already been proposed for speaker-modelling [5, 6] and speaker-adaptation [9] but not for speaking-style modelling. To achieve this we suggest to re-embed synthesised utterances that human annotators have tuned acoustically to match a target speaking style. After re-embedding the annotated utterances we aim to tune a fixed bank of style tokens [2]. This could not only further understanding the latent prosody space but also align it with human perception improving the quality of style generation.

[1] [Towards end-to-end prosody transfer for expressive speech synthesis with tacotron](http://proceedings.mlr.press/v80/skerry-ryan18a.html)

[2] [Style tokens: Unsupervised style modeling, control and transfer in end-to-end speech synthesis](http://proceedings.mlr.press/v80/wang18h.html?ref=https://githubhelp.com)

[3] [Daft-exprt: Robust prosody transfer across speakers for expressive speech synthesis](https://arxiv.org/abs/2108.02271)

[4] [Copycat: Many-to-many fine-grained prosody transfer for neural text-to-speech](https://arxiv.org/abs/2004.14617)

[5] [Perceptual-similarity-aware deep speaker representation learning for multi-speaker generative modeling](https://ieeexplore.ieee.org/iel7/6570655/9289074/09354556.pdf)

[6] [DNN-based Speaker Embedding Using Subjective Inter-speaker Similarity for Multi-speaker Modeling in Speech Synthesis](https://arxiv.org/pdf/1907.08294)

[7] [Fastspeech: Fast, robust and controllable text to speech](https://proceedings.neurips.cc/paper/2019/hash/f63f65b503e22cb970527f23c9ad7db1-Abstract.html)

[8] [Effective use of variational embedding capacity in expressive end-to-end speech synthesis](https://arxiv.org/abs/1906.03402)

[9] [Human-in-the-loop Speaker Adaptation for DNN-based Multi-speaker TTS](https://arxiv.org/abs/2206.10256)