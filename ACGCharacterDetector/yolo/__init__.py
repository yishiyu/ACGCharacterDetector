from .. import app
import torch

YOLO_PATH = app.config['YOLO_PATH']
MODEL_PATH = app.config['MODEL_PATH']


# 加载 yolov5 模型,选用最小的模型
model = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local')
model.eval()

app.logger.debug('acgmodel loaded')

__all__ = ['model']