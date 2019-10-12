# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:02:19 2019

@author: Asus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
 
# 讀取 CSV File
data = pd.read_csv('titanic.csv')  
data = np.array(data.values)
np.random.shuffle(data) 
for i in range(data.shape[1]-1):
    data[:,i+1]=(data[:,i+1] - np.mean(data[:,i+1]))/np.std(data[:,i+1])
#data[:,6]=(data[:,6]-np.mean(data[:,6]))/np.std(data[:,6])
#calculation function

#active function
def sigmoid(z):
    #z=np.where(z>=0,z+0.001,-0.001*z)
    #z=np.where(z>=1,0.999,z)
    #return np.matrix(z)
    return 1 / (1 + np.exp(-z))

#deviation of active function
def sigmoid_gradient(z):
    #np.where(z>=0,1,0.001)
    #return np.matrix(z)
    return np.multiply(sigmoid(z), (1 - sigmoid(z))) 

#forward propagation
def forward_propagate(X, theta1, theta2):
    a1 = np.insert(X, 0, 1, axis=1)

    w1 = a1.T
    z2 = theta1.dot(w1)
    #z2 = (z2 - np.mean(z2))/np.std(z2)
    z2 = z2.T
    
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    
    w2 = a2.T
    z3 = theta2.dot(w2)
    #z3 = (z3 - np.mean(z3))/np.std(z3)
    z3 = z3.T

    h  = sigmoid(z3)

    return a1, z2, a2, z3, h

#error_function
def error_function(theta1, theta2, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # compute the cost
    errorvalue = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log2(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log2(1 - h[i,:]))
        errorvalue += np.sum(first_term - second_term)
        
    errorvalue = errorvalue / (2*m)
    errorvalue += (float(learning_rate) / (2*m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    return errorvalue

#backword propagation
def backword_propagate(theta1, theta2, input_size, hidden_size, num_labels, X, y):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    #Write codes here
    theta1_grad=np.zeros(theta1.shape)
    theta2_grad=np.zeros(theta2.shape)

    #print(y[:,k].shape,sigmoid_gradient(z3[:,k]).shape,a2[:,j].shape)
    for k in range(num_labels):
        for j in range(hidden_size+1):
            theta2_grad[k][j]+=np.sum(np.multiply(np.multiply((y[:,k]-h[:,k]),sigmoid_gradient(z3[:,k])),a2[:,j]))
           
    #print(theta2[i,k].shape,y[:,k].shape,sigmoid_gradient(z3[:,k]).shape,a1[:,j].shape,a2[:,j].shape,)
    for k in range(num_labels):
        for j in range(hidden_size):
            for i in range(input_size+1):
                theta1_grad[j][i]+=np.sum(np.multiply(np.multiply(np.multiply(theta2[k,j+1]*(y[:,k]-h[:,k]),
                           sigmoid_gradient(z3[:,k])),sigmoid_gradient(z2[:,j])),a1[:,i]))

    theta2_grad = theta2_grad/m
    theta1_grad = theta1_grad/m
    
    return theta1_grad, theta2_grad

def change_to_hotcode(data):
    hotcode_data = np.array([data, data])
    hotcode_data = hotcode_data.T
    i=0
    for i in range(len(hotcode_data)):
        if hotcode_data[i,1]==0:
            hotcode_data[i,1]=1
        else:
            hotcode_data[i,1]=0
    return hotcode_data

#設計一個7*100*2的DNN
# initial setup
input_size = 6
hidden_size = 40
num_labels = 2
learning_rate = 1
mini_batch_size = 200
epoch = 100

# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
m = len(data[0:800,0])
data_x = data[0:800,1:7]
data_y = change_to_hotcode(data[0:800,0])
test_x = data[800:892,1:7]
test_y = change_to_hotcode(data[800:892,0])
#data_x=np.delete(data_x, 5, 1)
#test_x=np.delete(test_x, 5, 1)


# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))) #(20, 7)
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1)))) #(2, 21)

accuracy_before=0
accuracy_history=[]
errorvalue_history=[]
test_accuracy_history=[]

for i in range(epoch):
    a=random.randint(0,799)
    for j in range(int(m/mini_batch_size)):
        if(a+j*mini_batch_size>=800):
            a=a+j*mini_batch_size-800
        else:
            a=a+j*mini_batch_size
        if(a+mini_batch_size>=800):
            x=data_x[a:800,:]
            x=np.concatenate((x,data_x[0:a+mini_batch_size-800,:]), axis=0)
            y=data_y[a:800,:]
            y=np.concatenate((y,data_y[0:a+mini_batch_size-800,:]), axis=0)
        else:
            x=data_x[a:a+mini_batch_size,:]
            y=data_y[a:a+mini_batch_size,:]

        #forward
        a1, z2, a2, z3, h = forward_propagate(x, theta1, theta2)
        y_pred = np.array(np.argmin(h, axis=1))
        correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y[:,0])]
        accuracy = (sum(map(int, correct)) / float(len(correct)))
        errorvalue=error_function(theta1, theta2, x, y, learning_rate)
        
        #back
        theta1_grad, theta2_grad = backword_propagate(theta1, theta2, input_size, hidden_size, num_labels, x, y)
        theta1 = theta1+learning_rate*theta1_grad
        theta2 = theta2+learning_rate*theta2_grad
        if(accuracy_before>accuracy and learning_rate>0.01):
            learning_rate/=(1+0.1/((m/mini_batch_size)**1.5))
        accuracy_before=accuracy
    
    #test
    a1, z2, a2, z3, h = forward_propagate(test_x, theta1, theta2)
    test_y_pred = np.array(np.argmin(h, axis=1))
    test_correct = [1 if a == b else 0 for (a, b) in zip(test_y_pred, test_y[:,0])]
    test_accuracy = (sum(map(int, test_correct)) / float(len(test_correct)))
    test_errorvalue=error_function(theta1, theta2, test_x, test_y, learning_rate)
    
    print("learning_rate = ",learning_rate)
    print('accuracy = {0}%'.format(accuracy * 100))
    print("errorvalue = ",errorvalue)
    print('test accuracy = {0}%'.format(test_accuracy * 100))
    print("test errorvalue = ",test_errorvalue)
    print('\n')
    accuracy_history.append(1-accuracy)
    errorvalue_history.append(errorvalue)
    test_accuracy_history.append(1-test_accuracy)

plt.plot(accuracy_history)
plt . xlabel( "Number of epochs" ) 
plt . ylabel( "Error rate" )
plt.show()

plt.plot(errorvalue_history)
plt . xlabel( "Number of epochs" ) 
plt . ylabel( "average of cross entropy" )
plt.show()

plt.plot(test_accuracy_history)
plt . xlabel( "Number of epochs" ) 
plt . ylabel( "Error rate of test" )
plt.show()

live_unlive = np.array([[3,0,30,0,0,30],[1,1,70,1,1,10]])
print(live_unlive.shape)
a1, z2, a2, z3, h = forward_propagate(live_unlive, theta1, theta2)
test_y_pred = np.array(np.argmin(h, axis=1))
test_correct = [1 if a == b else 0 for (a, b) in zip(test_y_pred, test_y[:,0])]
test_accuracy = (sum(map(int, test_correct)) / float(len(test_correct)))
print(test_correct,test_accuracy)
