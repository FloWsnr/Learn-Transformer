# Learn Transformer

## Install

```bash
conda create -n learn_former python=3.11
conda install pytorch torchtext pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytest
pip install datasets, tokenizers
pip install -e .
```


## Learnings

- The transformer does not predict only the next word, but the entire sequence of words. That is, the output is not a single probability distribution, but a sequence of probability distributions, one for each word in the output sequence.
