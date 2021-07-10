# FashionMNIST-MLP-CNN

## Introduction
The aim of the project is to build a deep learning model that recognizes the fashion items of the Fashion-MNIST data set. To do this, we are going to build two different architectures (an MLP and a CNN architecture). Firstly, we discuss the problem and then we deploy the deep learning models. In the sequel of this report, challenges, problems, experimental results and demos will be reported. Finally, we will compare the two models and we will choose the optimal one.

## Fashion Item Recognition
As zalandoresearch - fashion-mnist repository refer, Fashion-MNIST is a dataset of Zalando's article image. The dataset consists of 60000 training images and 10000 test images. Each image represents a cloth that the deep learning models aim to find. The images are 28x28 grayscale associated with the label from 10 classes. These classes are:

|Label  | Item-Class|
| ----- | -----|
|0 |	T-shirt/top |
|1 |	Trouser |
|2 |	Pullover |
|3 |	Dress |
|4 |	Coat |
|5 |	Sandal |
|6 |	Shirt |
|7 |	Sneaker |
|8 |	Bag |
|9 |	Ankle boot |

We are going to import data as tensorflow fashion-mnist shows. Then we use the the train_test_split sklearn's function to create a set that we will use it in for the validation of the model. Finally, we have 3 sets (train, validation and test) with the following shapes:
| Set | Shale |
| ---- | ---- |
| x_train | (48000,28,28) |
| x_val | (12000,28,28) |
| x_test | (10000,28,28) |

![alt text](https://github.com/anmatrapazis/FashionMNIST-MLP-CNN/blob/main/im1.png)
