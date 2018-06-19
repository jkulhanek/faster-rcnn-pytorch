import torch
import torch.optim as optim
import torch.nn as nn
from loss import RPNLoss
import model


rpn_learning_rate = 0.01
rpn_epochs = 20


def build_rpn(**kwargs):
    backbone = model.vgg16(**kwargs)
    rpn_head = model.RPNHead()
    network = nn.Sequential(backbone, rpn_head)
    return network

def train_rpn():
    net = build_rpn(pretrained = True)
    optimizer = optim.SGD(net.parameters(), lr=rpn_learning_rate)
    criterion = criterion = RPNLoss()

    for epoch in range(rpn_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataset.trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
