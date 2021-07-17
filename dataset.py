import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL  import Image

class MyDataset(Dataset):

    def __init__(self):
        super().__init__()

        self.img_paths = os.listdir(r"./dataset/pics")
        self.img_paths.sort()

        label_paths = os.listir(r"./dataset/labels") 
        label_paths.sort()

        self.labels = []
        for label_path in label_paths:
            label_path = os.path.join(r"./dataset/labels",label_path)
            temp = []
            with open(label_path,"r") as f:
                for cell in f.read()[2:-1]:
                    temp.append(float(cell))

            self.labels.append(temp)        

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            trnasforms.Resize((224, 224)),
            transforms.Normalize(0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
            transforms.RandomErasing(),
        ])


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_path = os.path.join(r"./dataset/pics", img_path)

        img = Image.open(img_path)
        img = self.transforms(img)

        label = self.labels[index]
        label = torch.tensor(label)

        return img, label
    
    def __len__(self):
        return len(self.label)
