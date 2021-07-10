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

Lets have a look on images and labels.
![alt text](https://github.com/anmatrapazis/FashionMNIST-MLP-CNN/blob/main/im1.png)

## MLP model
### Model Description
Our model will be a keras.Sequential model consisted of a Flatten layers. This layer will help us to flatten the shape of the input. Then a dropout layer to drop randomly units in different layers. Then, we connect some dense hidden layers following by a dropout layers (dense-dropout-dense-drop-etc.) Finally, we connect with a final dense layer in order to provide the output we want. Here a to_categorical vector that shows with 1 the predicted label. A general idea would be to create an architecture with a number of hidden layers that follow general-to-specific reasoning. To be more specific the first hidden layer to have more hidden units an layer by layer the hidden units to be reduced until the final dense layer with exactly 10 hidden units.)  To make this process more optimal we will tune the model by doing the hyperparameter (hp) tuning with the help of keras-tuner. The goal is to tune the hidden layers, the hidden units, the dropout values and the learning rate for the optimizer.
At this stage we set the following hp strategy:
#### Hp Tuning
|Layer | Tags | Notes|
|----|----|----|
| dropout of the flatten layer| Choose from min_value = 0 until max_value = 0.3 with step =0.1 | We don't want to lose much units of the training data at the training process |
| #of hidden layers | Choose from 1 to 4 hidden layers | Randomly, time reasonable selected values
| hidden Dense layer | Choose a number between [512,256,128,64,32] | |
| hidden Dropout layer | Choose from min_value = 0 until max_value = 0.4 with step =0.1| |
| Learning rate | Choose a number between [1e-2,1e-3] | Choosing a lower value in this step would make the process slower. |

> For the fine-tuning process, we use the keras-tuner RandomSearch

After 5 trials and 2 executions per trail we have the following best architectures after the randomsearch hyperparameter tuning.  For the optimization we choose adam optimizer since it is provide better accuracy and loss than SGD optimizer. (Eustace Dogo et all DOI:DOI:10.1109/CTEMS.2018.8769211)
Note: The results taken using the keras set_seed in order to get and compare reproducible results.

### MLP with Dropouts layers:

