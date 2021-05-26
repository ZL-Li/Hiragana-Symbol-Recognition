# Hiragana Symbol Recognition

## Introduction

Our mission is to recognize handwritten Hiragana symbols.

The dataset to be used is Kuzushiji-MNIST or KMNIST for short.
The paper describing the dataset is available [here](https://arxiv.org/pdf/1812.01718.pdf).

Significant changes occurred to the language when Japan reformed their education system in 1868, and the majority of Japanese today cannot read texts published over 150 years ago.This paper presents a dataset of handwritten, labeled examples of this old-style script (Kuzushiji).
Along with this dataset, however, they also provide a much simpler one, containing 10 Hiragana characters with 7000 samples per class. This is the dataset we will be using.

Progressively more complex (and accurate) models are made.

### NetLin

`NetLin` model only computes a linear function of the pixels in the image, followed by log softmax.

Run the code by typing:
```
python3 kuzu_main.py --net lin
```

It will download the dataset if running for the first time.

It will show the final accuracy and confusion matrix after running. The final accuracy is approximately 70%.

Note that the rows of the confusion matrix indicate the target character,
while the columns indicate the one chosen by the network.
(0="o", 1="ki", 2="su", 3="tsu", 4="na", 5="ha", 6="ma", 7="ya", 8="re", 9="wo").
More examples of each character can be found [here](http://codh.rois.ac.jp/kmnist/index.html.en").

### NetFull

`NetFull` is a fully connected 2-layer network, using tanh at the hidden nodes and log softmax at the output node.

Run the code by typing:

```
python3 kuzu_main.py --net full
```

The final accuracy can achieve 85%, which is better than the first model.

### NetConv

`NetConv` is a convolutional network, with two convolutional layers plus one fully connected layer, all using relu activation function, followed by the output layer.

Run the code by typing:

```
python3 kuzu_main.py --net conv
```

The network can achieve 94% accuracy on the test set, which is the best of three.