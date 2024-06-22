# Learn Transformer

My attempt to understand the transformer model (attention is all you need) in-depth.
Feel free to reuse the code, but be aware that this is just a learning project and not a production-ready code.

I'm happy to get feedback on mistakes etc, so feel free to open an issue or a pull request.


## Install

```bash
conda create -n learn_former python=3.11
conda install pytorch torchtext pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytest
pip install datasets, tokenizers
pip install -e .
```


## Learnings

- The transformer does not predict only the next token, but the entire sequence of tokens. That is, the output is not a single probability distribution, but a sequence of probability distributions, one for each token in the input sequence. In other words, the transformer simultaneously learns to predict token 2 given token 1, token 3 given tokens 1 and 2, and so on.
- Masking is more complex than I thought.
    - First of all, the padding mask is used to prevent the model from looking at the padding tokens. This padding mask must have the length of the key sequence, i.e. which keys should be ignored.
    - The second mask is the look-ahead mask, which is used to prevent the model from cheating by looking at the future words in the sequence. It is only used in the decoder, while the encoder is allowed to look at the entire input sequence. This mask is a lower triangular matrix, which is applied to the scores (before the softmax) in the self-attention mechanism.
- Multi-head attention are simple matrix tricks to reshape the key, value and query matrices. When calculating the attentions, the head dimensions are handled similar to a batch dimension, i.e. the attention is calculated for all heads at once, but separatly.
- For training, the input to the decoder is trimmed by one token, i.e. the last token is removed. This is because the model is trained to predict the next token given the previous tokens, and the last token is not used to predict anything.
- Similarly, for the cross-entropy loss, the target is trimmed by one token, i.e. the first token is removed, i.e. the sequence of targets is shifted to the left by one token. Thereby, traget and the output of the decoder match up. (decoder output is the prediction of the next token, and the target is the next token)
