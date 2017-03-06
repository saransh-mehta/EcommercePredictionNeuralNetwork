# building a feed forward network

import numpy as np
import matplotlib.pyplot as plt

#generatiing some data

Dno=350
x1=np.random.randn(Dno,2) + np.array([2,-2])   #cloud centerred at 0,-2
x2=np.random.randn(Dno,2) + np.array([2,2])    #cloud centerred at 2,2
x3=np.random.randn(Dno,2) + np.array([-2,-2])  #cloud centerred at -2,-2

x=np.vstack([x1,x2,x3])                   #stacking th 3 arrays vertically one below other (row wise)
y=np.array([0]*Dno+[1]*Dno+[2]*Dno)       #y=[0,0,...1,1..,2,2..]

#print(y)                               0->x1 1->x2 2->x3 hot-encoding

#plotting the data for viewing

#plt.scatter(x[:,0],x[:,1],)
#plt.show()

# details about input layer, hid and output layer 

inLay=2
hidLay=3
outLay=3

#creating weights and biases

w1=np.random.randn(inLay,hidLay) #weights for first input layer
b1=np.random.randn(hidLay)

w2=np.random.randn(hidLay,outLay) #weights for 2nd output layer
b2=np.random.randn(outLay)

def forward(x,w1,b1,w2,b2):

    #applyin sigmoid to input so as to get proper positive value
    
    z=1/(1+np.exp(-x.dot(w1)-b1))
    #plt.scatter(z[:,0],z[:,1])
    #plt.show()

    a=z.dot(w2) + b2

    #applying softmax to output layer to get probability
    
    expa=np.exp(a)
    output=expa/ expa.sum(axis=1, keepdims=True)
    #print(output)
    #print(len(output))
    return output

def classificationRate(output,outClassify):  #clas rate can be thought of
                                        # no. of current hit/total no of tries
    total=0
    correct=0
   
    for i in range(len(output)):
        total+=1
        if (y[i]==outClassify[i]):    #remember y is like [0,0,0..n times,1,1..n tims,2,2..n tims]

            correct+=1
    return (correct/total)
        
                   

result=forward(x,w1,b1,w2,b2)

outClassify=np.argmax(result, axis=1)   #argmax gives the index of max value along the axis
#print(outClassify)                                        #we do ths as to get max prob is of which class

assert(len(outClassify)==len(result))                                        #n then compare the obtained index 0,1 or 2
                                        #to actual result i.e, y
print("classification rate is : ",classificationRate(result,outClassify)) #the classifier should give0.33 as equally distributed

