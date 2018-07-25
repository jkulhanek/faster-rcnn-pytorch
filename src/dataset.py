import torch
import torchvision
import model
import model_utils as utils
import torchvision.transforms as transforms


DATASET_PATH = "/datasets/pascalvoc/VOC2007/JPEGImages"
DATASET_TRAIN_TARGET = "/datasets/pascalvoc/annotations/pascal_train2007.json"

DATASET_PATH = "D:\\datasets\\pascalvoc\\VOC2007\\JPEGImages"
DATASET_TRAIN_TARGET = "D:\\datasets\\pascalvoc\\annotations\\pascal_train2007.json"

def target_transform(target):
    # Ignore ignored
    target = [x for x in target if x['ignore'] != 1]
    target_count = len(target)
    boxes = torch.empty((target_count, 4,), requires_grad = False)
    labels =torch.empty((target_count,), requires_grad = False)

    for i,elem in enumerate(target):
        boxes[i] = torch.tensor(elem['bbox'])
        labels[i] = elem['category_id']
    return {
        "boxes": boxes,
        "labels": labels
    }

def make_dataset(dataset_root_path = DATASET_PATH, dataset_train_target = DATASET_TRAIN_TARGET):
    return torchvision.datasets.CocoDetection(DATASET_PATH, DATASET_TRAIN_TARGET, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), 
        target_transform=transforms.Lambda(target_transform))