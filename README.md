## Problem Description
Train LSTM neural network for captcha recognition using TensorFLow.  

## LSTM Model

### Data Collection

`data_gen_pro.py` aims to generate 10000 training captchas and 64 test captchas for training and testing. Captchas are all constructed by 4 lowercases.  

### Model Architecture
`get_batch` is to determine `x` and `y` for captchas. `x` is shaped to [60,160] while `y` is a one-dimention array sized 26.

`computational_graph_lstm` is the LSTM Model. The model uses 128 hidden units and 2 hidden layers.  


### Training
Training parameters are:
```
batch_size = 64
learning_rate = 0.001
iteration = 10000
```


## Result
Trainin result is:
```
Test Accuracy: 0.351563
```

*This accuracy indictaes to accuracy of recognition of characters but not a complete captcha.* 

## Contributors
* 2014010193 HowieLiang
* 2014012182 lujq96
* 2014012170 zxp14
* 2015011544 Jiawei