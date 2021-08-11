from logging import debug
from .. import app
import json
import torch
from torchvision import transforms

YOLO_PATH = app.config['YOLO_PATH']
YOLO_MODEL = app.config['YOLO_MODEL']
RESNET18_MODEL = app.config['RESNET18_MODEL']
RESNET18_LABELS = app.config['RESNET18_LABELS']
RESNET18_TRANS = app.config['RESNET18_TRANS']

# 加载 yolov5 模型,选用最小的模型
yolov5 = torch.hub.load(YOLO_PATH, 'custom', path=YOLO_MODEL, source='local')
yolov5.eval()

# 加载 resnet18 模型
resnet18_model = torch.load(RESNET18_MODEL)
resnet18_model.to('cpu')
resnet18_model.eval()

class_names = []
translate = {}
# 加载标签
with open(RESNET18_LABELS,'r',encoding='utf-8') as file:
    for line in file.readlines():
        class_names.append(line.strip('\n'))
with open(RESNET18_TRANS, 'r', encoding='utf-8') as file:
    translate = json.load(file)

transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def resnet18(image):
    """给resnet18包装一个数据预处理器
    """
    temp = transformer(image)
    outputs =  resnet18_model(
        temp[None,...]
    )
    _, preds = torch.max(outputs, 1)
    name = class_names[preds[0]]
    return name,translate[name]


app.logger.debug('acgmodel loaded')

__all__ = ['yolov5', 'resnet18']
