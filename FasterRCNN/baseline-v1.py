import pandas as pd
import numpy as np
import os
import re
import torch, cv2
from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

MY_TEST = True
num_epochs = 1
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cpu'

DIR_INPUT = '../input'
DIR_WEIGHTS = '../models'
WEIGHTS_FILE = f'{DIR_WEIGHTS}/fasterrcnn_resnet50_fpn.pth'
# f表示格式化字符串
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
# train_df.shape

train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    # 寻找字符串x中所有匹配这种格式的字符，[0-9]+ 表示小数点前至少有一个数字
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

# .apply: Apply a function along an axis of the DataFrame.
# np.stack默认axis=0,把数组竖着拼
train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
# Drop specified labels from rows or columns.
# inplace=True, return None
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

# .unique()去掉重复值，返回numpy.ndarray
image_ids = train_df['image_id'].unique()
# 共3373个图片
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]

# isin返回布尔索引, train_df 是包含image_id, weight,height,x,y,w,h的7列DataFrame，最后665个是valid，前面的是train
valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]

# TODO：WheatDataset
class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        if MY_TEST is True:
            image_id = "00333207f"
        records = self.df[self.df['image_id'] == image_id]
        '''
        reads the image with RGB colors but no transparency channel. 
        This is the default value for the flag when no value is provided as the second argument for cv2.imread().
        '''
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        # cv2.cvtColor() method is used to convert an image from one color space to another. It returns an image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        # narray: shape=(55, 4) (左上角，右下角)
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        # area:shape=(55,), 每个bbox的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class， records.shape[0]=55
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        # sample和target一样，也是一个字典
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }

            # 这里传入一个字典，self.transforms对字典中key=image的值进行处理
            # 这里如果有翻转等操作，字典中的boxes也会一起变化，和image保持同步
            sample = self.transforms(**sample)
            image = sample['image']
            # permute 更换维度，stack堆叠向量
            # 下面这句不知道啥意思，直接torch.Tensor(sample['bboxes'])结果相同
            # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            target['boxes'] = torch.Tensor(sample['bboxes'])
        # image(c,w,h)
        # target:'boxes','labels','image_id'=tensor([index]),'area','iscrowd'
        # image_id 为图片文件名
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
# TODO :CV2画出边界框样例
def draw_img(sample,boxes):
    boxes = boxes.cpu().numpy().astype(np.int32)
    sample = sample.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0), 3)

    ax.set_axis_off()
    ax.imshow(sample)

# Albumentations
def get_train_transform():
    return A.Compose([
        # 对图片以0.5的概率垂直翻转，A.Flip默认是垂直翻转
        A.Flip(0.5),
        # 转为Tensor，注意这里如果输入类型为unit8将会自动÷256
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# TODO model
# load a model; pre-trained on COCO
'''
Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.
The input to the model is expected to be a list of tensors, each of shape [C, H, W], 
one for each image, and should be in 0-1 range. Different images can have different sizes.
The behavior of the model changes depending if it is in training or evaluation mode.
During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:

~ boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x between 0 and W and values of y between 0 and H

~ labels (Int64Tensor[N]): the class label for each ground-truth box

The model returns a Dict[Tensor] during training, 
containing the classification and regression losses for both the RPN and the R-CNN.
During inference, the model requires only the input tensors, and returns the post-processed 
predictions as a List[Dict[Tensor]], one for each input image.
 
The fields of the Dict are as follows:
~ boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x between 0 and W and values of y between 0 and H

~ labels (Int64Tensor[N]): the predicted labels for each image

~ scores (Tensor[N]): the scores or each prediction

Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
'''
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # 1 class (wheat) + background

'''
1 - Finetuning from a pretrained model
Let’s suppose that you want to start from a model pre-trained on COCO and want to finetune 
it for your particular classes. Here is a possible way of doing it:
下面两步都是为了在COCO预训练集上Finetune自己的数据
'''
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# TODO：Averager
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())

# split the dataset in train and test set
# torch.randperm：Returns a random permutation of integers from 0 to n - 1.
# indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# TODO：train
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# torch.optim.lr_scheduler.StepLR 每隔step_size对学习率乘上gamma
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()

    for images, targets, image_ids in train_data_loader:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")

# TODO：valid
images, targets, image_ids = next(iter(valid_data_loader))

images = list(img.to(device) for img in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
sample = images[1].permute(1,2,0).cpu().numpy()

model.eval()
cpu_device = torch.device("cpu")

outputs = model(images)
outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

torch.save(model.state_dict(), WEIGHTS_FILE)
