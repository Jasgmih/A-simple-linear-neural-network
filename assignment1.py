import numpy as np
import torch
import matplotlib.pyplot as plt
m = 8  # number of examples
l1, l2, l3 = 8, 3, 8  # number of neuros in layer0, layer1 and layer2
alpha = 0.3  # learning rate

# derivate of sigmoid function
def sigmoid_derivate(x): 
    return torch.mul(torch.sigmoid(x), 1-torch.sigmoid(x))

data= torch.eye(l1)
y = torch.eye(l3)

# initial weights and biases
w1 = torch.randn(l2,l1)
w2 = torch.randn(l3, l2)
b1 = np.zeros(shape=(l2, 1))           # vector of shape (3, 1)
b2 = np.zeros(shape=(l3, 1))           # vector of shape (8, 1)

# b0 = torch.randn(1)
# b1 = torch.randn(1)

losses = []
epoches = 20000
for epoch in range(epoches):
    
    # forward propagation
    # z1 = torch.mm(w1, data).add(b0)
    z1 = torch.mm(w1, data) + b1
    a1 = torch.sigmoid(z1) # activate in layer1 (hidden layer)
    # z2 = torch.mm(w2, a1).add(b1)
    z2 = torch.mm(w2, a1) + b2
    a2 = torch.sigmoid(z2) # activate in layer2(output layer)

    # loss function
    loss = torch.sum((a2-data)**2)
    losses.append(loss.item())

    # back propagation
    # layer2 (output layer)
    da2 = a2.sub(y).mul(2)   # derivative of activate in layer2
    dz2 = torch.mul(da2, sigmoid_derivate(z2)) 
    dw2 = torch.mm(dz2, a1.T).div(m)
    db2 = (1 / m) * np.sum(dZ2)   # bias term

    
    # layer1 (hidden layer)
    da1 = torch.mm(w2.T, dz2)
    dz1 = torch.mul(da1, sigmoid_derivate(z1))
    dw1 = torch.mm(dz1, data.T).div(m)
    db1 = (1 / m) * np.sum(dZ1)   # bias term


    # update weights and biases
    w1 -= alpha*dw1
    b1 = b1 - learning_rate * db1

    
    w2 -= alpha*dw2
    b2 = b2 - learning_rate * db2


    print("epoch: "+ str(epoch) + ", loss: "+ str(loss.item()))

# test
z1 = torch.mm(w1, data) + b1
a1 = torch.sigmoid(z1)
z2 = torch.mm(w2, a1) + b2
a2 = torch.sigmoid(z2) 
print(a2)
print(a2.argmax(dim=0)) # print the index of max value in each line

# plt.xlabel("epoch") 
# plt.ylabel("loss") 
plt.plot(np.arange(epoches), losses)
plt.show()


"""
Haven't added bias yet
"""
