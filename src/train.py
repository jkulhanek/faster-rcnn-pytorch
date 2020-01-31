import torch
import torch.optim as optim
import torch.nn as nn
from loss import RPNLoss
import model
from dataset import make_dataset
from torch.utils.data import Dataset, DataLoader


rpn_learning_rate = 0.001
rpn_iterations = 80000
batch_size = 1


def build_rpn(**kwargs):
    backbone = model.vgg16(**kwargs)
    rpn_head = model.RPNHead()
    network = nn.Sequential(backbone, rpn_head)
    return network

def train_rpn(**kwargs):
    transformed_dataset = make_dataset(**kwargs)
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    net = build_rpn(pretrained = True)    
    optimizer = optim.SGD([param for param in net.parameters() if param.requires_grad], lr=rpn_learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60000], gamma=0.1)
    criterion = criterion = RPNLoss()

    iteration = 0
    rpn_epochs = rpn_iterations // len(transformed_dataset)
    print("Total number of epoch is {}".format(rpn_epochs))
    for epoch in range(rpn_epochs):        
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            scheduler.step()
            # get the inputs
            inputs, labels = data

            # Extends the labels with the image size
            labels["size"] = inputs.size()[2:]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2 == 1:
            #if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
            iteration += 1

def test_rpn_training(**kwargs):
    transformed_dataset = make_dataset(**kwargs)
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    net = build_rpn(pretrained = True)    
    optimizer = optim.SGD([param for param in net.parameters() if param.requires_grad], lr=rpn_learning_rate)
    criterion = RPNLoss()

    rpn_epochs = rpn_iterations // len(transformed_dataset)
    data = next(iter(dataloader))

    print("Total number of epoch is {}".format(rpn_epochs))
    for epoch in range(rpn_epochs):        
        running_loss = 0.0
        # get the inputs
        inputs, labels = data

        # Extends the labels with the image size
        labels["size"] = inputs.size()[2:]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
            (epoch + 1, epoch + 1, running_loss / 2000))
        running_loss = 0.0


if __name__ == "__main__":
    test_rpn_training()
