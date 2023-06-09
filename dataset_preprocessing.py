import os
import shutil
import random  
import string  
import torch
import math
import numpy as np
from torchvision import datasets
import albumentations as A   
import albumentations.pytorch as album_pytorch


class Paths:
    """Class for keeping paths to directories and files"""
    pandora_18k : str = '/home/noonmare/Activity/Diploma/Datasets/Pandora18k/'
    pandora_18k_resized : str = '/home/noonmare/Activity/Diploma/Datasets/Pandora18k_resized/'

class Dataset:
    """Class for keeping dataset of paintings"""
    def __init__(self, path_to_the_dataset) -> None:
        self.path_to_the_dataset = path_to_the_dataset

    @property
    def train_path(self) -> str:
        """Path to directory with train samples"""
        return self.path_to_the_dataset + 'Train/'
    
    @property
    def val_path(self) -> str:
        """Path to directory with validation samples"""
        return self.path_to_the_dataset + 'Validation/'
    
    @property
    def test_path(self) -> str:
        """Path to directory with test samples"""
        return self.path_to_the_dataset + 'Test/'

    @property
    def number_of_classes(self) -> int:
        """return number of classes in dataset"""
        return len(next(os.walk(self.train_path))[1])
    
    @property
    def classes(self) -> list:
        """return list of classes in dataset"""
        return next(os.walk(self.train_path))[1]
    
    @property
    def train_samples_per_class(self) -> dict:
        """return dict of classes with frequencies for train set"""
        styles_data = {}
        for cl in next(os.walk(self.train_path))[1]:
            styles_data[cl] = len(os.listdir(self.train_path + cl))
        return styles_data

    @property
    def val_samples_per_class(self) -> dict:
        """return dict of classes with frequencies for validation set"""
        styles_data = {}
        for cl in next(os.walk(self.val_path))[1]:
            styles_data[cl] = len(os.listdir(self.val_path + cl))
        return styles_data

    @property
    def test_samples_per_class(self) -> dict:
        """return dict of classes with frequencies for train set"""
        styles_data = {}
        for cl in next(os.walk(self.test_path))[1]:
            styles_data[cl] = len(os.listdir(self.test_path + cl))
        return styles_data

    @property
    def train_len(self) -> int:
        """return number of samples in train set"""
        sum = 0
        for cl in next(os.walk(self.train_path))[1]:
            sum += len(os.listdir(self.train_path + cl))
        return sum

    @property
    def val_len(self) -> int:
        """return number of samples in validation set"""
        sum = 0
        for cl in next(os.walk(self.val_path))[1]:
            sum += len(os.listdir(self.val_path + cl))
        return sum

    @property
    def test_len(self) -> int:
        """return number of samples in test set"""
        sum = 0
        for cl in next(os.walk(self.test_path))[1]:
            sum += len(os.listdir(self.test_path + cl))
        return sum

    

    def train_val_test_split(self, val_size=0.15, test_size=0.1) -> None:
        """split dataset to train and test parts"""
        train_path = self.train_path
        val_path = self.val_path
        test_path = self.test_path

        styles = self.classes

        if not os.path.exists(train_path) and not os.path.exists(val_path) and not os.path.exists(test_path):
            os.makedirs(train_path)
            os.makedirs(val_path)
            os.makedirs(test_path)

            print('Train, validation and test directories are created')

            for style in styles:

                if style == "Train" or style == "Validation" or style == "Test":
                    continue

                try:

                    train_style_path = train_path + '/' + style
                    val_style_path = val_path + '/' + style
                    test_style_path = test_path + '/' + style
                    os.makedirs(train_style_path)
                    os.makedirs(val_style_path)
                    os.makedirs(test_style_path)

                    home_style_path = self.path_to_the_dataset + '/' + style

                    style_paintings = os.listdir(home_style_path)
                    style_quantity = len(style_paintings)

                    val_part = int(style_quantity*val_size)
                    test_part = int(style_quantity*test_size)

                    for painting in style_paintings[:test_part]:
                        fr = os.path.join(home_style_path, painting)
                        to = os.path.join(test_style_path, painting)
                        shutil.move(fr, to)

                    for painting in style_paintings[test_part:test_part+val_part]:
                        fr = os.path.join(home_style_path, painting)
                        to = os.path.join(val_style_path, painting)
                        shutil.move(fr, to)

                    for painting in style_paintings[test_part+val_part:]:
                        fr = os.path.join(home_style_path, painting)
                        to = os.path.join(train_style_path, painting)
                        shutil.move(fr, to)
                except:
                    print(f'Problem with {style}')
                else:
                    shutil.rmtree(home_style_path)

        else:
            print('Train, validation and test directories already exist')

    def train_val_test_split_advanced(self, val_size=0.15, test_size=0.2) -> None:
        """split dataset to train and test parts"""
        train_path = self.train_path
        val_path = self.val_path
        test_path = self.test_path

        styles = self.classes

        if not os.path.exists(train_path) and not os.path.exists(val_path) and not os.path.exists(test_path):
            os.makedirs(train_path)
            os.makedirs(val_path)
            os.makedirs(test_path)

            print('Train, validation and test directories are created')

        for style in styles:

            if style == "Train" or style == "Validation" or style == "Test":
                continue

            try:

                train_style_path = train_path + '/' + style
                val_style_path = val_path + '/' + style
                test_style_path = test_path + '/' + style

                if not os.path.exists(train_style_path) and not os.path.exists(val_style_path) and not os.path.exists(test_style_path):
                    os.makedirs(train_style_path)
                    os.makedirs(val_style_path)
                    os.makedirs(test_style_path)

                home_style_path = self.path_to_the_dataset + '/' + style + '/'

                authors = next(os.walk(home_style_path))[1]

                for author in authors:
                    source = home_style_path + author + '/'
                    style_paintings = os.listdir(source)
                    style_quantity = len(style_paintings)

                    val_part = math.ceil(style_quantity*val_size)
                    test_part = math.ceil(style_quantity*test_size)

                    print(style, style_quantity-(val_part+test_part), val_part, test_part)

                    for painting in style_paintings[:test_part]:
                        file_name = os.path.join(source, painting)
                        try:
                            shutil.move(file_name, test_style_path)
                        except:
                        # Error when in file with the same name already exists in destination directory
                            print(f'{file_name} already exists')
                            random_str = ''.join((random.choice(string.ascii_lowercase) for x in range(10)))
                            new_file_name = source + random_str + '.jpg'
                            os.rename(file_name, new_file_name)
                            shutil.move(new_file_name, test_style_path)

                    for painting in style_paintings[test_part:test_part+val_part]:
                        file_name = os.path.join(source, painting)
                        try:
                            shutil.move(file_name, val_style_path)
                        except:
                        # Error when in file with the same name already exists in destination directory
                            print(f'{file_name} already exists')
                            random_str = ''.join((random.choice(string.ascii_lowercase) for x in range(10)))
                            new_file_name = source + random_str + '.jpg'
                            os.rename(file_name, new_file_name)
                            shutil.move(new_file_name, val_style_path)

                    for painting in style_paintings[test_part+val_part:]:
                        file_name = os.path.join(source, painting)
                        try:
                            shutil.move(file_name, train_style_path)
                        except:
                        # Error when in file with the same name already exists in destination directory
                            print(f'{file_name} already exists')
                            random_str = ''.join((random.choice(string.ascii_lowercase) for x in range(10)))
                            new_file_name = source + random_str + '.jpg'
                            os.rename(file_name, new_file_name)
                            shutil.move(new_file_name, train_style_path)
            except:
                print(f'Problem with {style}')
            print(f'style {style} finished')

class Transforms:
    def __init__(self, transforms: A.Compose):
            self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))

class Dataloader:
    def __init__(self, model, train_path, val_path, test_path, batch_size) -> None:
        
        """ image_datasets = {
        'train' : datasets.ImageFolder(root=train_path, transform=model.transforms['train']),
        'validation' : datasets.ImageFolder(root=val_path, transform=model.transforms['validation']),  
        'test' : datasets.ImageFolder(root=test_path, transform=model.transforms['test'])  
        }  """

        image_datasets = {
        'train' : datasets.ImageFolder(root=train_path, transform=Transforms(transforms=model.transforms_alb['train'])),
        'validation' : datasets.ImageFolder(root=val_path, transform=Transforms(transforms=model.transforms_alb['validation'])),  
        'test' : datasets.ImageFolder(root=test_path, transform=Transforms(transforms=model.transforms_alb['test']))  
        } 
    
        dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=8),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=batch_size, shuffle=True, num_workers=8),
        'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True, num_workers=8)
                   }
    
        dataset_sizes = {set_type: len(image_datasets[set_type]) for set_type in ['train', 'validation', 'test']}

        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes

class Dataloader_parts:
    def __init__(self, model, lu_path, ld_path, ru_path, rd_path, c_path, batch_size) -> None:
        image_datasets = {
        'lu' : datasets.ImageFolder(lu_path, model.transforms['train']),
        'ld' : datasets.ImageFolder(ld_path, model.transforms['train']),
        'ru' : datasets.ImageFolder(ru_path, model.transforms['train']),
        'rd' : datasets.ImageFolder(rd_path, model.transforms['train']),
        'c' : datasets.ImageFolder(c_path, model.transforms['train'])
        }
    
        dataloaders = {
        'lu' : torch.utils.data.DataLoader(image_datasets['lu'], batch_size=batch_size, shuffle=False, num_workers=8),
        'ld' : torch.utils.data.DataLoader(image_datasets['ld'], batch_size=batch_size, shuffle=False, num_workers=8),
        'ru' : torch.utils.data.DataLoader(image_datasets['ru'], batch_size=batch_size, shuffle=False, num_workers=8),
        'rd' : torch.utils.data.DataLoader(image_datasets['rd'], batch_size=batch_size, shuffle=False, num_workers=8),
        'c' : torch.utils.data.DataLoader(image_datasets['c'], batch_size=batch_size, shuffle=False, num_workers=8)
        }
    
        dataset_sizes = {set_type: len(image_datasets[set_type]) for set_type in dataloaders.keys()}

        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes


def Pandora18k_process(Pandora18k_dataset : Dataset) -> None:
    """Move files from authors' directories to styles' directories"""
    Pandora18k_styles = Pandora18k_dataset.classes

    for style in Pandora18k_styles:
        style_path = Paths.pandora_18k + style + '/'
        authors = next(os.walk(style_path))[1]
        for author in authors:
            source = style_path + author + '/'
            files = os.listdir(source)
            try:
                for file in files:
                    file_name = os.path.join(source, file)
                    try:
                        shutil.move(file_name, style_path)
                    except:
                        # Error when in file with the same name already exists in destination directory
                        print(f'{file_name} already exists')
                        random_str = ''.join((random.choice(string.ascii_lowercase) for x in range(10)))
                        new_file_name = source + random_str + '.jpg'
                        os.rename(file_name, new_file_name)
                        shutil.move(new_file_name, style_path)
            except:
                print(f'Problem with author : {author}, style : {style}')
            else:
                shutil.rmtree(source)
            

        print(f"{style} : Files Moved")

if __name__ == "__main__":

    Pandora18k_dataset = Dataset(path_to_the_dataset=Paths.pandora_18k)

    #Pandora18k_process(Pandora18k_dataset)

    Pandora18k_dataset.train_val_test_split_advanced()
