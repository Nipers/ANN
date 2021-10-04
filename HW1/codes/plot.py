import numpy as np 
from matplotlib import pyplot as plt 

acc_test = "accuracy/test/"
acc_train = "accuracy/train/"
loss_test = "loss/test/"
loss_train = "loss/train/"

x1 = np.arange(1,201)
file = open(loss_train + "0.02.txt", "r")
y1 = []
line = file.readline()
i = 0
while line:
    y1.append(float(line))
    line = file.readline()
# print(y1)
file.close()
x2 = np.arange(5,205, 5)
file = open(loss_test + "0.02.txt", "r")
y2 = []
line = file.readline()
i = 0
while line:
    y2.append(float(line))
    i += 1
    line = file.readline()
file.close()
plt.title('Loss') 
plt.xlabel("epoch") 
plt.ylabel("loss") 
plt.plot(x1,y1, label = "train") 
plt.plot(x2,y2, label = "test") 
plt.legend()
plt.savefig("T_0.02.png")
# plt.show()