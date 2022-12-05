from time import time
import torch
import torch.nn as nn
import numpy as np



class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        pass

    def forward(self):
        pass

class InceptionV3Trainer(nn.Module):
    def __init__(self,InceptionV3Net,trainloader,testloader,epochs=10,device=None):
        super(InceptionV3Trainer, self).__init__()
        self.model = InceptionV3Net
        
        # use SGD optimizer with momentum and 0.0005 learning rate base TransferLearning approach paper
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0005, momentum=0.9)
        # use cosine loss function base on paper
        # cosine loss function has better result on small datasets
        self.loss_function = nn.CrossEntropyLoss()
        self.train_loss = np.zeros(epochs)
        self.train_acc = np.zeros(epochs)
        self.test_loss = np.zeros(epochs)
        self.test_acc = np.zeros(epochs)

        self.epochs = epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.cuda()

        self.trainloader = trainloader
        self.testloader = testloader


    def forward(self,x):
        # get output from inceptionV3
        # FcClassifierOutput , AuxLogitsClassifierOutput = self.InceptionV3Net(x)
        # return FcClassifierOutput , AuxLogitsClassifierOutput
        x = self.model(x)
        return x

    def evaluate(self):
        self.eval()
        size = len(self.testloader.dataset)
        num_batches = len(self.testloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.testloader:
                X = X.cuda()
                y = y.type(torch.LongTensor)
                y = y.cuda()
                pred = self.model(X)
                test_loss += self.loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        return correct , test_loss

    def train_one_epoch(self):
        self.train()
        size = len(self.trainloader.dataset)
        num_batches = len(self.trainloader)
        train_loss,correct = 0,0
        for batch, (X, y) in enumerate(self.trainloader):
          # Compute prediction and loss
          X = X.cuda()
          y = y.type(torch.LongTensor)
          y = y.cuda()
          pred = self.model(X)

          # Backpropagation
          self.optimizer.zero_grad()
          self.loss_function(pred,y).backward()
          self.optimizer.step()

        with torch.no_grad():
          for X,y in self.trainloader:
            X = X.cuda()
            y = y.type(torch.LongTensor)
            y = y.cuda()
            pred = self.forward(X)
            train_loss += self.loss_function(pred,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        train_loss/=num_batches
        correct/=size
        return correct , train_loss

    def fit(self):
        first_time = time()
        for epoch in range(self.epochs):
          train_accuracy , train_cost = self.train_one_epoch()
          test_accuracy , test_cost = self.evaluate()
          self.train_acc[epoch] = train_accuracy
          self.train_loss[epoch] = train_cost
          self.test_acc[epoch] = test_accuracy
          self.test_loss[epoch] = test_cost
          second_time = time()
        #   self.countTime.append(second_time-first_time)
          print(f"Epoch {epoch+1} in {int(second_time-first_time)} , Train Accuracy: {(100*train_accuracy):>0.1f}%, Avg train loss: {train_cost:>8f} , Test Accuracy: {(100*test_accuracy):>0.1f}%, Avg test loss: {test_cost:>8f} \n")
        torch.save(self.state_dict(),f=f"{self.__class__.__name__}.pt")

    def getStats(self):
        return self.train_acc,self.train_loss,self.test_acc,self.test_loss

    # def getTime(self):
        # return self.countTime


