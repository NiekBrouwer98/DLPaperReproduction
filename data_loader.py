# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:19:14 2022

@author: Tibbe Lukkassen
"""

import torch
import torchvision as tv
import torchvision.transforms.functional as TF


def pad(img, size_max=256):
    """
    Pads images to the specified size (height x width). 
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)
        
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
        
    return TF.pad(
        img,
        (pad_left, pad_top, pad_right, pad_bottom),
        fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))

#image = np.load('./CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg')
    # transform images
transforms_train = tv.transforms.Compose([
   tv.transforms.Resize(size=255, max_size=256),
   tv.transforms.Lambda(pad),
   tv.transforms.RandomCrop((224, 224)),
   tv.transforms.RandomHorizontalFlip(),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
])

transforms_test = tv.transforms.Compose([
   tv.transforms.Resize(size=255, max_size=256),
   tv.transforms.Lambda(pad),
   tv.transforms.CenterCrop((224, 224)),
   tv.transforms.RandomHorizontalFlip(),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
])

ds_train = tv.datasets.ImageFolder('CUB_100_train/images', transform=transforms_train)
ds_test = tv.datasets.ImageFolder('CUB_100_test/images', transform=transforms_test)

trainloader = torch.utils.data.DataLoader(ds_train, batch_size=(10), shuffle=True)
testloader = torch.utils.data.DataLoader(ds_test)

for epochs in range(1):
    for data in trainloader:
        x=data[0]
        y=data[1]
        #outputs = model(x,y)
        print (y)
        break

#dataiter = iter(testloader)
#data = next(dataiter)
#data2 = next(dataiter)
#data3 = next(dataiter)

#dataiter2 = iter(trainloader) 
#datatrain = next(dataiter2) #1
#datatrain2= next(dataiter2) #2
#datatrain2= next(dataiter2) #3
#datatrain2= next(dataiter2) #4
#datatrain2= next(dataiter2) #5
#datatrain2= next(dataiter2) #6
#datatrain2= next(dataiter2) #7
#datatrain2= next(dataiter2) #8
# datatrain2= next(dataiter2) #9
# datatrain2= next(dataiter2) #10

# transform = tv.transforms.ToPILImage()
# img2 = transform(datatrain2[0][0])
# img2.show()
# img = transform(data[0][0])
# img2 = transform(data2[0][0])
# img3 = transform(datatrain[0][0])
# img4 = transform(datatrain2[0][0])
# img.show()
# img2.show()
# img3.show()
# img4.show()