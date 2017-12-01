'''
Created on Dec 1, 2017

@author: Amar Viswanathan
'''
import numpy as np
import math
from numpy import vstack, hstack
import matplotlib.pyplot as plt
import sys
import random
from scipy.special import expit
import copy


np.set_printoptions(threshold=sys.maxint)


def sigmoid(x):
    """ returns the sigmoid of x """ 
    output = 1/(1 + np.exp(-x))
    return output

def sigmoid_output_derivative(output):
    """ calculates the derivative part for backpropagation """ 
    return np.dot(output,(1 - output))

def one_hot_encode(y,n,num_classes):
    """ Multiple classes exist in the training data, so this function does the one-hot-encoding """ 
    yTrain = np.zeros((n,num_classes))
    index = 0
    for toBeEncoded in y:
        num =  int(toBeEncoded)
        yTrain[index,num-1] = 1
#         print str(num) + " , " + str(yTrain[index])
        index = index + 1
    return yTrain
    
def init_W(n,d):
    """ Initialize random weights between -0.1 and 0.1 """ 
#     random.seed(0.001)
    W = np.zeros((n,d))
    for i in range(n):
        for j in range(d):
            W[i,j] = random.uniform(-0.1,0.1)
    return W


def ReLU(output):
    """ Relu activation """ 
    X = [max(0,x) for x in output]
    return X

def ReLU_deriv(output):   
    """ Relu derivative for backpropagation """ 
    X = [1 if x > 0 else 0 for x in output]
    x = np.asarray(X)
    return x

def feedforward(xi,W,Wdash,ReLUVar):
    """ Calculate the feedforward values """ 
    # Layer 1 is the hidden layer
    net_layer_1 = np.dot(xi,W)
    if(ReLUVar):
        out_layer_1 = np.asarray(ReLU(net_layer_1))
    else:
        out_layer_1 = expit(net_layer_1)
    out_layer_1 = np.append(out_layer_1,[1])
    net_layer_2 = np.dot(out_layer_1,Wdash)
    out_layer_2 = expit(net_layer_2)
    return out_layer_1,out_layer_2

def updateW(W,eta, out_layer, delta_layer):
    """ Update weights after backpropagation """ 
    (n,d) = W.shape
    for i in range(n):
        for j in range(d):
            W[i,j] = W[i,j] - (eta * (out_layer[i] * delta_layer[j]))
    return W
    
def ReLU_Delta(deriv, delta_layer_2,W_dash):
    """ Calculate the deltas if relu is used """ 
    (n,d) = W_dash.shape
    delta_layer_1 = np.zeros(Nh+1)
    for j in range(n):
        sum = 0
        for k in range(d):
            sum = sum + (delta_layer_2[k] * W_dash[j,k])
        delta_layer_1[j] = deriv[j] * sum
    return delta_layer_1
    

def backpropagation(yhat_i,y_i,xi,out_layer_1,eta,W,W_dash,ReLUVar):
    # for output layer
    #oj - tj term
    error_layer_2 = yhat_i - y_i
    # calculating delta = (oj - tj) * oj * (1-oj)
    delta_layer_2 = np.dot(error_layer_2,sigmoid_output_derivative(yhat_i))

    # for hidden layer
    if(ReLUVar):  
        deriv = ReLU_deriv(out_layer_1)
        delta_layer_1 = ReLU_Delta(deriv, delta_layer_2, W_dash)
#         delta_layer_1 = np.dot(deriv, np.dot(delta_layer_2, W_dash.T))
    else:
        delta_layer_1 = np.dot(sigmoid_output_derivative(out_layer_1),np.dot(delta_layer_2,W_dash.T))
    W_dash = updateW(W_dash, eta, out_layer_1, delta_layer_2)
    W      = updateW(W,eta,xi,delta_layer_1)
  

    return W,W_dash


   
# trainFileName = "shuttle.trn.txt"
# testFileName = "shuttle.tst.txt"

# trainFileName = "iris-versicolor.txt"
# testFileName = "iris-versicolor.txt"
  
# Nh = 5
# eta = 0.01
# epoch = 10

""" Total number of classes in the training set""" 
num_classes = 7

trainFileName = sys.argv[1]
testFileName = sys.argv[2]
Nh = int(sys.argv[3])
eta  = float(sys.argv[4])
epoch = int(sys.argv[5])
option = sys.argv[6]
print "Option is " + str(option)
if(option == "sigmoid"):
    ReLUVar = False
elif(option == "relu"):
    ReLUVar = True

# print "The value of ReLUVar is " + str(ReLUVar)

train = np.loadtxt(trainFileName,dtype='float',delimiter=',')
test = np.loadtxt(testFileName,dtype='float',delimiter=',')

yTrain = train[:,-1]
xTrain = train[:,:-1]
# Old shape
(n,d) = xTrain.shape
xTrain = np.append(xTrain, np.ones((n,1)),1)
# New shape after appending 1s
(n,d) =  xTrain.shape

# 
# y = np.zeros((n,7))
# print y

yTrain = one_hot_encode(yTrain, n, num_classes)
# print yTrain

# W_b = np.random.uniform(low=-0.1, high = 0.1, size = (d,Nh))
# W_dash = np.random.uniform(low=-0.1,high = 0.1, size = (Nh,7))


# W = np.random.uniform(low=-0.1, high = 0.1, size = (d,Nh))
# W_dash = np.random.uniform(low=-0.1,high = 0.1, size = (Nh+1,num_classes))

W = init_W(d,Nh)    # layer 1 weights i.e. input to hidden
W_dash = init_W(Nh+1,num_classes)   # hidden to output weights




for e in range(epoch-1):
#     print "Epoch number " + str(e + 1) + " \n"
#     print "\n------------------------------"
    random_order = range(n)
    np.random.shuffle(random_order)
    for xi in random_order:
        out_i = feedforward(xTrain[xi],W, W_dash,ReLUVar)
        yhat_i = out_i[1]
        E = 0.5 * np.linalg.norm(yhat_i - yTrain[xi])**2
        if E > 0:
#             print out_i[0]
            Wout = backpropagation(yhat_i, yTrain[xi], xTrain[xi],out_i[0], eta, W, W_dash,ReLUVar)
            W = Wout[0]
            W_dash = Wout[1]

print "For the hidden layer " + option + " \n"
print "The weight vector of shape " + str(W.shape) + " (along with bias) for the input to hidden layer is : \n"
print W
print "--------------------------------------------------------------------"
print "The weight vector of shape " + str(W_dash.shape) + " along with bias for hidden to output layer is : \n"
print W_dash
print "--------------------------------------------------------------------"

xTest = test[:,:-1]
yTest = test[:,-1]
# Old shape
(nTest,dTest) = xTest.shape
xTest = np.append(xTest, np.ones((nTest,1)),1)
# New shape after appending 1s
(nTest,dTest) =  xTest.shape
yTest = one_hot_encode(yTest, nTest, num_classes)       

order = range(nTest)
count = 0
for xiTest in order:
    outTest = feedforward(xTest[xiTest], W, W_dash,ReLUVar)
    out = outTest[1]
#     print str(np.argmax(yTest[xiTest])) + " : " + str(np.argmax(np.argmax(out)))
    if(np.argmax(yTest[xiTest]) == np.argmax(out)):
        count = count + 1
    
print "Accuracy on the test set is "  + str((float)(count)/(nTest))


