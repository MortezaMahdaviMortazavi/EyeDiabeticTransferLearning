import os
import pandas as pd
import numpy as np
import torch
from skimage import io
#import ToTensor
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir, transform=None,classification="normal"):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.classification = classification

        self.classes = self.data["level"].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        binary_label = None
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = io.imread(img_name+".jpeg")
        label = self.data.iloc[idx, 1]

        # convert to binary classification
        if label == 0:
            binary_label = 0
        else:
            binary_label = 1

        if self.transform:
            # image = np.array(image)
            image = self.transform(image)

        if self.classification == "binary":
            label = binary_label
    


        return image, label

# level
# 0    25808
# 2     5291
# 1     2442
# 3      873
# 4      708
# dtype: int64

def DataCollectionForTrain(dataset,csv_path):
    csv_file = pd.read_csv(csv_path)
    X_batch_train_holder = torch.zeros((10000,3,300,300))
    y_batch_train_holder = torch.zeros((10000))
    X_batch_test_holder = torch.zeros((2000,3,300,300))
    y_batch_test_holder = torch.zeros((2000))
    train_index = 0
    test_index = 0
    is_zero_filled = False
    is_one_filled = False
    is_two_filled = False
    is_three_filled = False
    is_four_filled = False

    # now define booleans that control test dataset
    is_zero_test_filled = False
    is_one_test_filled = False
    is_two_test_filled = False
    is_three_test_filled = False

    each_label_count_train = np.zeros((5,1))
    each_label_count_test = np.zeros((5,1))
    # now fill the test dataset

    for i in range(len(dataset)):
        X,y = dataset[i]
        if y == 0 and not is_zero_filled:
            X_batch_train_holder[train_index] = X
            y_batch_train_holder[train_index] = int(y)
            train_index += 1
            
            each_label_count_train[0] += 1
            if each_label_count_train[0] == 5000:
                is_zero_filled = True
            continue

        elif y == 1 and not is_one_filled:
            X_batch_train_holder[train_index] = X
            y_batch_train_holder[train_index] = int(y)
            train_index += 1
            
            each_label_count_train[1] += 1
            if each_label_count_train[1] == 2500:
                is_one_filled = True
            continue

        elif y == 2 and not is_two_filled:
            X_batch_train_holder[train_index] = X
            y_batch_train_holder[train_index] = int(y)
            train_index += 1
            
            each_label_count_train[2] += 1
            if each_label_count_train[2] == 1500:
                is_two_filled = True
            continue

        elif y == 3 and not is_three_filled:
            X_batch_train_holder[train_index] = X
            y_batch_train_holder[train_index] = int(y)
            train_index += 1
            
            each_label_count_train[3] += 1
            if each_label_count_train[3] == 500:
                is_three_filled = True
            continue

        elif y == 4 and not is_four_filled:
            X_batch_train_holder[train_index] = X
            y_batch_train_holder[train_index] = int(y)
            train_index += 1
            
            each_label_count_train[4] += 1
            if each_label_count_train[4] == 500:
                is_four_filled = True
            continue


        return X_batch_train_holder,y_batch_train_holder

def DataCollectionForTest():
    dataset = get_dataset()
    X_batch_train_holder,X_batch_test_holder,y_batch_train_holder,y_batch_test_holder = data_splittor(dataset)
    return X_batch_train_holder,X_batch_test_holder,y_batch_train_holder,y_batch_test_holder


def ImageTransformer():
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((300,300)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.3199, 0.2240, 0.1609],
            # std=[0.3020, 0.2183, 0.1741],)
    ])
    return transform



def get_dataset(transofmer=ImageTransformer()):
    dataset = CustomDataset(csv_path='../trainLabels.csv', root_dir='../train/images_resized_300/', transform=transofmer)
    return dataset


def data_splittor(X,y,test_size=0.2): # based on sklearn train_test_split method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def main():
    print("Nothing done here")


if __name__ == '__main__':
    main()         