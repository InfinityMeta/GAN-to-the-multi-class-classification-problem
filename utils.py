import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from dataset_preprocessing import Paths, Dataset

IMAGE_PARTS = ['lu', 'ld', 'ru', 'rd', 'c']
TRAIN_PARTS = ["Train", "Validation", "Test"]
ds = Dataset(Paths.pandora_18k)
CLASSES = ds.classes

class MyDataset(Dataset):
 
  def __init__(self, df, num_classes):
    
    x=df.iloc[:,0:6*num_classes].values
    y=df.iloc[:,-1].apply(lambda x: x-1).values   
 
    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.x_train = self.x_train.reshape((self.x_train.shape[0], 1, self.x_train.shape[1]))
    self.y_train=torch.tensor(y)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]
  
class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )
  
def train_parts_folders(train_parts=TRAIN_PARTS, path=Paths.pandora_18k, classes=CLASSES):
    for tp in train_parts:
        folder_path = path + tp 
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for cl in classes:
            class_path = folder_path + '/' + cl
            os.mkdir(class_path)

def image_parts_folders(train_parts=TRAIN_PARTS, image_parts=IMAGE_PARTS, path=Paths.pandora_18k, classes=CLASSES):
    for tp in train_parts:
        for ip in image_parts:
            folder_path = path + tp + '_' + ip
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            for cl in classes:
                class_path = folder_path + '/' + cl
                os.mkdir(class_path)

def lu_cropp(im_path):
    im = Image.open(im_path).convert("RGB")
    width, height = im.size
    left = 0
    top = 0
    right = width // 2
    bottom = height // 2
    im_cropped = im.crop((left, top, right, bottom))
    return im_cropped

def ld_cropp(im_path):
    im = Image.open(im_path).convert("RGB")
    width, height = im.size
    left = 0
    top = height // 2
    right = width // 2
    bottom = height
    im_cropped = im.crop((left, top, right, bottom))
    return im_cropped

def ru_cropp(im_path):
    im = Image.open(im_path).convert("RGB")
    width, height = im.size
    left = width // 2
    top = 0
    right = width 
    bottom = height // 2
    im_cropped = im.crop((left, top, right, bottom))
    return im_cropped

def rd_cropp(im_path):
    im = Image.open(im_path).convert("RGB")
    width, height = im.size
    left = width // 2
    top = height // 2
    right = width 
    bottom = height
    im_cropped = im.crop((left, top, right, bottom))
    return im_cropped

def c_cropp(im_path):
    im = Image.open(im_path).convert("RGB")
    width, height = im.size
    left = width // 4
    top = height // 4
    right = left + width // 2
    bottom = top + height // 2
    im_cropped = im.crop((left, top, right, bottom))
    return im_cropped

CROPP_FUNCS = {
    "lu" : lu_cropp,
    "ru" : ru_cropp,
    "c" : c_cropp,
    "ld" : ld_cropp,
    "rd" : rd_cropp
} 

def resize(im_path, input_shape=(299, 299)):
    im = Image.open(im_path).convert('RGB')
    im_resized = im.resize(input_shape)
    return im_resized

def resize_images(train_parts=TRAIN_PARTS, path_fr=Paths.pandora_18k, path_fr_resized=Paths.pandora_18k_resized):
    for tp in train_parts:
        path_from = path_fr + tp + '/'
        for cl in ds.classes:
            cl_path_from = path_from + cl + '/'
            imgs_names = os.listdir(cl_path_from)
            for im_name in imgs_names:
                im_path = cl_path_from + im_name
                resized_im_path =  path_fr_resized + tp + '/' + cl + '/'  + 'resized_' + im_name
                resized_im = resize(im_path)
                resized_im.save(resized_im_path)

def cut_images(train_parts=TRAIN_PARTS, path=Paths.pandora_18k, cropp_funcs=CROPP_FUNCS):
    for tp in train_parts:
        path_from = path + tp + '/'
        for cl in ds.classes:
            cl_path_from = path_from + cl + '/'
            imgs_names = os.listdir(cl_path_from)
            for im_name in imgs_names:
                im_path = cl_path_from + im_name
                for cr_type, cr_func in cropp_funcs.items():
                    cr_im_path =  path + tp + '_' + cr_type + '/' + cl + '/' + cr_type + '_' + im_name
                    cropped_im = cr_func(im_path)
                    cropped_im.save(cr_im_path)






