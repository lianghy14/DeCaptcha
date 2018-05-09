#-*- coding:utf-8 -*

import tensorflow as tf
import os
import random
import numpy as np
from PIL import Image

path = os.getcwd()  #directory
captcha_path = path + '/images'  
test_path = path + '/tests'
output_path = path + '/result/result.txt'   

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

batch_size = 64  #size of batch
time_steps = 60   
n_input = 160  
image_channels = 1  # number of channerls
captcha_num = 4 # number of characters
n_classes = len(alphabet) #number of classes

learning_rate = 0.001   #learning rate for adam
num_units = 128   #hidden LSTM units
layer_num = 2   #number of layers
iteration = 10000   #number of iterations

def get_batch(data_path = captcha_path, is_training = True):
    target_file_list = os.listdir(data_path)    

    batch = batch_size if is_training else len(target_file_list) 
    batch_x = np.zeros([batch, time_steps, n_input]) 
    batch_y = np.zeros([batch, captcha_num, n_classes]) 

    for i in range(batch):
        file_name = random.choice(target_file_list) if is_training else target_file_list[i]
        img = Image.open(data_path + '/' + file_name) 
        img = np.array(img)
        if len(img.shape) > 2:
            img = np.mean(img, -1)  #(60,160,3) to (60,160,1)
            img = img / 255 
        batch_x[i] = img

        label = np.zeros(captcha_num * n_classes)
        for num, char in enumerate(file_name.split('.')[0]):
            index = num * n_classes + char2index(char)
            label[index] = 1
        label = np.reshape(label,[captcha_num, n_classes])
        batch_y[i] = label
    return batch_x, batch_y

def char2index(c):
    k = ord(c)
    index = -1
    if k >= 97 and k <= 122:
        index = k - 97
    if index == -1:
        raise ValueError('No Map')
    return index

def index2char(k):
    # k = chr(num)
    index = -1
    if k >= 0 and k < 26: 
        index = k + 97
    if index == -1:
        raise ValueError('No Map')
    return chr(index)

def computational_graph_lstm(x, y, batch_size = batch_size):

    #weights and biases of appropriate shape to accomplish above task
    out_weights = tf.Variable(tf.random_normal([num_units,n_classes]), name = 'out_weight')
    out_bias = tf.Variable(tf.random_normal([n_classes]),name = 'out_bias')

    #model construction
    lstm_layer = [tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True) for _ in range(layer_num)] #two-layer lstm
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layer, state_is_tuple = True)  

    init_state = mlstm_cell.zero_state(batch_size, tf.float32)

    outputs = list() 
    state = init_state
    with tf.variable_scope('RNN'):
        for timestep in range(time_steps):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = mlstm_cell(x[:, timestep, :], state) 
            outputs.append(cell_output)

    prediction_1 = tf.nn.softmax(tf.matmul(outputs[-4],out_weights)+out_bias)  
    prediction_2 = tf.nn.softmax(tf.matmul(outputs[-3],out_weights)+out_bias)  
    prediction_3 = tf.nn.softmax(tf.matmul(outputs[-2],out_weights)+out_bias)  
    prediction_4 = tf.nn.softmax(tf.matmul(outputs[-1],out_weights)+out_bias)  
    prediction_all = tf.concat([prediction_1, prediction_2, prediction_3, prediction_4],1)  
    prediction_all = tf.reshape(prediction_all,[batch_size, captcha_num, n_classes],name ='prediction_merge') 

    #loss_function
    loss = -tf.reduce_mean(y * tf.log(prediction_all),name = 'loss')
    #optimization
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name = 'opt').minimize(loss)
    #model evaluation
    pre_arg = tf.argmax(prediction_all,2,name = 'predict')
    y_arg = tf.argmax(y,2)
    correct_prediction = tf.equal(pre_arg, y_arg)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name = 'accuracy')

    return opt, loss, accuracy, pre_arg, y_arg

def train():

    # defining placeholders
    x = tf.placeholder("float",[None,time_steps,n_input], name = "x") #input image placeholder
    y = tf.placeholder("float",[None,captcha_num,n_classes], name = "y")  #input label placeholder

    # computational graph
    opt, loss, accuracy, pre_arg, y_arg = computational_graph_lstm(x, y)

    saver = tf.train.Saver()  
    init = tf.global_variables_initializer()   

    with tf.Session() as sess:
        sess.run(init)
        iter = 1
        while iter < iteration:
            batch_x, batch_y = get_batch()
            sess.run(opt, feed_dict={x: batch_x, y: batch_y}) 
            if iter %100==0:
                los, acc, parg, yarg = sess.run([loss, accuracy, pre_arg, y_arg],feed_dict={x:batch_x,y:batch_y})
                print("For iter ",iter)
                print("Accuracy ",acc)
                print("Loss ",los)
                if iter % 1000 ==0:
                    print("predict arg:",parg[0:10])
                    print("yarg :",yarg[0:10])
                print("__________________")
                # if acc > 0.95:
                #     print("training complete, accuracy:", acc)
                #     break
            iter += 1
        # test accuracy
        test_x, test_y = get_batch(data_path=test_path, is_training=False)
        print("Test Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y}))
        
if __name__ == '__main__':
    train()

