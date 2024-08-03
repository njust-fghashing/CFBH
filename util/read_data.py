import torch.utils.data as data
from torchvision import transforms
from util.dataset_fine import dataset,Food101,NABirds
def read_dataset(dataset_name,batch_size):
    if dataset_name == 'cub_bird':
        classes = 200
        data_dir = '/dataset/cub_bird/'
    elif dataset_name == 'vegfru':
        classes = 292
        data_dir = '/dataset/vegfru/'
    elif dataset_name == 'aircraft':
        classes = 100
        data_dir = '/dataset/aircraft/'
    elif dataset_name == 'food101':
        classes = 101
        data_dir = '/dataset/food101/'
    elif dataset_name == 'stanford_car':
        classes = 196
        data_dir = '/dataset/stanford_car/'
    elif dataset_name == 'nabirds':
        classes = 555
        data_dir = '/dataset/nabirds/'
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    test_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    #  train_transforms = transforms.Compose([
    #         transforms.Resize((256,256)),
    #         transforms.RandomCrop((224,224)),          
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])
    #      test_transforms = transforms.Compose([
    #         transforms.Resize((224,224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #      ])

    # }
    if dataset_name == 'food101':
        train_dataset = Food101(dataset, root_dir=data_dir, transforms = train_transforms, train =True)
        test_dataset = Food101(dataset ,root_dir=data_dir, transforms = test_transforms, test =True)
        base_dataset = Food101(dataset, root_dir=data_dir, transforms = test_transforms, train =True)
    elif dataset_name == 'nabirds':
        train_dataset = NABirds( root_dir=data_dir, transforms = train_transforms, train =True)
        test_dataset = NABirds(root_dir=data_dir, transforms = test_transforms, test =True)
        base_dataset = NABirds(root_dir=data_dir, transforms = test_transforms, train =True)
    else :
        train_dataset = dataset(dataset_name, root_dir=data_dir, transforms = train_transforms, train =True)
        test_dataset = dataset(dataset_name ,root_dir=data_dir, transforms = test_transforms, test =True) 
        base_dataset = dataset(dataset_name, root_dir=data_dir, transforms = test_transforms, train =True)
        

    
    train_dataloader= data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader= data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    base_dataloader = data.DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return classes,train_dataloader, test_dataloader, base_dataloader
