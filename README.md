# myGPT2

## Building a GPT-2 Model from scratch!

This repo is inspired by and follows along Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) youtube playlist.

In this repo, I built a GPT-2 clone based on the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and the official [GPT-2 repository](https://github.com/openai/gpt-2). This model is build using only decoder self-attention blocks and does not implement the encoder block with cross-attention architecture as shown in the GPT-2 paper.

The model in this repo has 124M parameters and uses the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) to train. Specifically, the FineWeb-Edu dataset with 10B tokens.

Evaluation was done using [Hellaswag](https://github.com/rowanz/hellaswag) and compared to the 124M parameter GPT-2 model and the 124M parameter GPT-3 model.

## Resources

- [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Hellaswag: Can A Machine Really Finish Your Sentence?](https://arxiv.org/pdf/1905.07830)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
