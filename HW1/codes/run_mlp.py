from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
models = []
losses = []
for i in range(0,6):
    if i < 3:#one layer
        model = Network()
        model.add(Linear('fc1', 784, 128, 0.01))
        if i == 0:
            model.add(Relu("r1"))
        if i == 1:
            model.add(Gelu("r1"))
        if i == 2:
            model.add(Sigmoid("r1"))
        model.add(Linear('fc1', 128, 10, 0.01))
        print(str(model))
        models.append(model)
    else:#two layer
        model = Network()
        model.add(Linear('fc1', 784, 256, 0.01))
        model.add(Relu("r1"))
        model.add(Linear('fc1', 256, 64, 0.01))
        if i == 3:
            model.add(Relu("r1"))
        if i == 4:
            model.add(Gelu("r1"))
        if i == 5:
            model.add(Sigmoid("r1"))
        model.add(Linear('fc1', 64, 10, 0.01))
        print(str(model))
        models.append(model)

losses.append(SoftmaxCrossEntropyLoss(name='loss'))
losses.append(EuclideanLoss(name='loss'))
losses.append(HingeLoss(name='loss'))


# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.003,
    'weight_decay': 0.0,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 200,
    'disp_freq': 50,
    'test_epoch': 5
}

for i in range(0, 6):
    for j in range(0, 3):
        index = i * 3 + j
        model = models[i]
        loss = losses[j]
        acc_train = "accuracy/train/"
        acc_test = "accuracy/test/"
        loss_train = "loss/train/"
        loss_test = "loss/test/"
        print(str(model) + str(loss))
        for epoch in range(config['max_epoch']):
            LOG_INFO('Training @ %d epoch...' % (epoch))
            l1, l2 = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
            fileName = loss_train + str(model) + str(loss)+".txt"
            file = open(fileName, "a")
            file.write(str(l1) + "\n")
            fileName = acc_train + str(model) + str(loss)+".txt"
            file = open(fileName, "a")
            file.write(str(l2) + "\n")
            if epoch % config['test_epoch'] == 0:
                LOG_INFO('Testing @ %d epoch...' % (epoch))
                l, t = test_net(model, loss, test_data, test_label, config['batch_size'])
                fileName = loss_test + str(model) + str(loss)+ ".txt"
                file = open(fileName, "a")
                file.write(str(l) + "\n")
                file.close()
                fileName = acc_test + str(model) + str(loss)+ ".txt"
                file = open(fileName, "a")
                file.write(str(t) + "\n")
                file.close()