# easy-VQA-Pytorch

This is a pytorch reimplementation of a simple Visual Question Answering (VQA) architecture, using the [easy-VQA](https://github.com/vzhou842/easy-VQA) dataset. The neural network architecture and preprocessing are basically the same with original implementation. We thank the author VictorZhou for releasing the code.

Methodology described in the official [blog post](https://victorzhou.com/blog/easy-vqa/). See [easy-VQA featured on the official VQA site](https://visualqa.org/external.html)!

## Usage

### Setup and Basic Usage

First, clone the repo and install the dependencies:

```shell
git clone https://github.com/sikaiyin/easy-VQA-Pytorch.git
pip install -r requirements.txt
```

To run the model,

```shell
python model.py
```

## Reference
easy-VQA-keras https://github.com/vzhou842/easy-VQA-keras

PyTorch MNIST https://github.com/pytorch/examples/tree/master/mnist
