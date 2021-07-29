from .. import app
import torch

YOLO_PATH = app.config['YOLO_PATH']
MODEL_PATH = app.config['MODEL_PATH']

app.logger.info('test')
app.logger.info(YOLO_PATH)
app.logger.info(MODEL_PATH)
app.logger.info('test')

# 加载 yolov5 模型,选用最小的模型
model = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local')

__all__ = ['model']