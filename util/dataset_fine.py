import os
import torch
import torch.utils.data as data
import PIL.Image as Image
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, extract_archive
from torchvision.datasets.folder import default_loader
class dataset(data.Dataset):
    def __init__(self, set_name, root_dir=None, transforms=None,train=False, test=False):
        self.root_path = root_dir
        self.train = train
        self.test = test
        self.transforms=transforms
        if self.train:
            self.train_anno = pd.read_csv(os.path.join(self.root_path, set_name+'_train.txt'), \
                                      sep=" ", \
                                      header=None, \
                                     names=['ImageName', 'label'])
            self.paths= self.train_anno['ImageName'].tolist()
            self.labels = self.train_anno['label'].tolist()
        if self.test:
            self.test_anno = pd.read_csv(os.path.join(self.root_path, set_name+'_test.txt'), \
                                      sep=" ", \
                                      header=None, \
                                     names=['ImageName', 'label'])
            self.paths= self.test_anno['ImageName'].tolist()

            self.labels = self.test_anno['label'].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)#PIL Image
        if self.test:
            img = self.transforms(img)
            label = self.labels[item]
            return img, label, item
        if self.train:
            img = self.transforms(img)
            label = self.labels[item]
            return img, label, item
    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

class Food101(data.Dataset):


    def __init__(self, set_name, root_dir=None, transforms=None, train=False, test=False):
        self.root_path = root_dir
        self.train = train
        self.test = test
        self.transforms = transforms
        image_class_labels = pd.read_csv(os.path.join(self.root_path, 'meta/labels.txt'), names=['target'])
        # print(image_class_labels)
        d = {}
        for i in range(len(image_class_labels)):
            image_class_labels['target'][i] = image_class_labels['target'][i].replace(' ', '_')
            image_class_labels['target'][i] = image_class_labels['target'][i].lower()
            d[image_class_labels['target'][i]] = i + 1
        images_train = pd.read_csv(os.path.join(self.root_path, 'meta/train.txt'), names=['filepath'])
        images_test = pd.read_csv(os.path.join(self.root_path, 'meta/test.txt'), names=['filepath'])
        train_images = []
        test_images = []
        for i in range(len(images_train)):
            train_images.append(images_train['filepath'][i] + '.jpg')
        for i in range(len(images_test)):
            test_images.append(images_test['filepath'][i] + '.jpg')
        # print(images_train[:10])
        label_list_train = []
        img_id_train = []
        for i in range(len(train_images)):
            label = train_images[i].split('/')[0]
            label_list_train.append(d[label])
            img_id_train.append(i + 1)
        images_train = []
        for i in range(len(train_images)):
            images_train.append([img_id_train[i], 'images/' + train_images[i], label_list_train[i]])

        images_train = pd.DataFrame(images_train, columns=['img_id', 'filepath', 'target'])
        k = len(train_images)
        label_list_test = []
        img_id_test = []
        for i in range(len(test_images)):
            label = test_images[i].split('/')[0]
            label_list_test.append(d[label])
            img_id_test.append(k + i + 1)
        images_test = []
        for i in range(len(test_images)):
            images_test.append([img_id_test[i], 'images/' + test_images[i], label_list_test[i]])
        images_test = pd.DataFrame(images_test, columns=['img_id', 'filepath', 'target'])
        train_data = images_train
        test_data = images_test
        if self.train:
            self.data=train_data['filepath'].to_numpy()
            self.targets=((train_data['target'] - 1).tolist())
        if self.test:
            self.data=test_data['filepath'].to_numpy()
            self.targets=((test_data['target'] - 1).tolist())

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_path, self.data[idx])).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.targets[idx], idx

class NABirds(VisionDataset):

    base_folder = 'images/'
    filename = 'nabirds.tar.gz'
    md5 = 'df21a9e4db349a14e2b08adfd45873bd'

    def __init__(self, root_dir=None, transforms=None, train=False, test=False):

        self.train=train
        self.test=test
        self.transform=transforms
        dataset_path = root_dir
        if not os.path.isdir(dataset_path):
            if not check_integrity(os.path.join(root_dir, self.filename), self.md5):
                raise RuntimeError('Dataset not found or corrupted.')
            extract_archive(os.path.join(root_dir, self.filename))
        self.root = os.path.expanduser(root_dir)
        self.loader = default_loader
        self.class_names = load_class_names(root_dir)
        self.class_hierarchy = load_hierarchy(root_dir)


        image_paths = pd.read_csv(os.path.join(root_dir, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        all_data = data.merge(train_test_split, on='img_id')
        all_data['filepath'] = 'images/' + all_data['filepath']
        all_data['target'] = all_data['target'].apply(lambda x: label_map[x])

        train_data = all_data[all_data['is_training_img'] == 1]
        # test_data = all_data[all_data['is_training_img'] == 0].iloc[5001:10551, :]
        test_data = all_data[all_data['is_training_img'] == 0]
        class_num = len(label_map)
        # Load in the train / test split
        if self.train:
            self.data=train_data['filepath'].to_numpy()
            self.targets=((train_data['target'] ).tolist())
        if self.test:
            self.data=test_data['filepath'].to_numpy()
            self.targets=((test_data['target'] ).tolist())

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx], idx


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents
