import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


def make_non_trainable(model):
    for param in model.parameters():
        param.requires_grad = False


def make_trainable(model):
    for param in model.parameters():
        param.requires_grad = True



def create_pretrained_inceptionV3():
    InceptionV3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    torch.save(InceptionV3, "inceptionV3.pt")


    InceptionV3.AuxLogits.fc = nn.Linear(768, 5)
    InceptionV3.fc = nn.Linear(2048, 5)

    # freeze convolution parts of inceptionV3
    make_non_trainable(InceptionV3.Conv2d_1a_3x3)
    make_non_trainable(InceptionV3.Conv2d_2a_3x3)
    make_non_trainable(InceptionV3.Conv2d_2b_3x3)
    make_non_trainable(InceptionV3.Conv2d_3b_1x1)
    make_non_trainable(InceptionV3.Conv2d_4a_3x3)
    make_non_trainable(InceptionV3.Mixed_5b)
    make_non_trainable(InceptionV3.Mixed_5c)
    make_non_trainable(InceptionV3.Mixed_5d)
    make_non_trainable(InceptionV3.Mixed_6a)
    make_non_trainable(InceptionV3.Mixed_6b)
    make_non_trainable(InceptionV3.Mixed_6c)
    make_non_trainable(InceptionV3.Mixed_6d)
    make_non_trainable(InceptionV3.Mixed_6e)
    # make_non_trainable(InceptionV3.Mixed_7a)
    # make_non_trainable(InceptionV3.Mixed_7b)
    # make_non_trainable(InceptionV3.Mixed_7c)

    make_trainable(InceptionV3.Mixed_7a)
    make_trainable(InceptionV3.Mixed_7b)
    make_trainable(InceptionV3.Mixed_7c)
    make_trainable(InceptionV3.AuxLogits)
    make_trainable(InceptionV3.fc)
    # set aug logit to false
    # InceptionV3.aux_logits = False
    
    def forward_imp(x):
        with torch.no_grad():
            x = InceptionV3.Conv2d_1a_3x3(x)
            x = InceptionV3.Conv2d_2a_3x3(x)
            x = InceptionV3.Conv2d_2b_3x3(x)
            x = InceptionV3.maxpool1(x)
            x = InceptionV3.Conv2d_3b_1x1(x)
            x = InceptionV3.Conv2d_4a_3x3(x)
            x = InceptionV3.maxpool2(x)
            x = InceptionV3.Mixed_5b(x)
            x = InceptionV3.Mixed_5c(x)
            x = InceptionV3.Mixed_5d(x)
            x = InceptionV3.Mixed_6a(x)
            x = InceptionV3.Mixed_6b(x)
            x = InceptionV3.Mixed_6c(x)
            x = InceptionV3.Mixed_6d(x)
            x = InceptionV3.Mixed_6e(x)
        

        x = InceptionV3.Mixed_7a(x)
        x = InceptionV3.Mixed_7b(x)
        x = InceptionV3.Mixed_7c(x)

        with torch.no_grad():
            x = InceptionV3.avgpool(x)
            x = torch.flatten(x, 1)
            x = InceptionV3.dropout(x)

        # x = InceptionV3.AuxLogits(x)
        x = InceptionV3.fc(x)
        return x    
    # assign model forward function to the new forward_imp function
    InceptionV3.forward = forward_imp
    return InceptionV3



def main():
    model = create_pretrained_inceptionV3()
    # test model on random data
    x = torch.randn(1,3,299,299)
    print(model(x))

if __name__ == "__main__":
    main()
