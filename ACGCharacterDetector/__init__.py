import io
from logging import log
from PIL import Image
from flask_cors import CORS
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)
CORS(app)

# 读取配置文件
import configparser
config = configparser.ConfigParser()
config.read("config.ini")

app.config['YOLO_PATH'] = config['YOLO']['yolo_path']
app.config['YOLO_MODEL'] = config['YOLO']['yolo_model']
app.config['RESNET18_MODEL'] = config['RESNET18']['resnet18_model']
app.config['RESNET18_LABELS'] = config['RESNET18']['resnet18_labels']
app.config['RESNET18_TRANS'] = config['RESNET18']['resnet18_translate_dict']

# 不可把这个调到前面,因为 acgmodel 的初始化需要配置信息
from .models import yolov5, resnet18

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """识别动漫人物头像位置并识别人物

    Returns:
        json: {'result':[
            [x,y,x,y,probability,name],
            ...
        ]}
    """
    if request.method == 'POST':
        # 读取传来的图片
        file = request.files['image']
        imagebytes = file.read()
        image = Image.open(io.BytesIO(imagebytes))

        boxes = __detect(image)
        result = __recognize(image, boxes)

        return jsonify(result)

    return jsonify([
        # [x,y,x,y,probability]
        {
            "box":[1,1,3,3,0.9],
            "name": "name",
            "trans": "trans"
        }
    ])


def __detect(image: Image):
    """识别图中头像的位置

    Args:
        image (Image): PIL.Image对象,被识别的图片

    Returns:
        list: 识别到的头像框位置[(x,y,x,y,probability,class),...]
    """
    boxes = yolov5(image).xyxyn[0]

    # app.logger.debug(type(box))
    # app.logger.debug(box)

    return boxes.tolist()


def __recognize(image: Image, boxes: list):
    """识别头像人物

    Args:
        image (Image): 原始图片
        boxes (list): 头像位置

    Returns:
        list: 识别到的头像框位置及人物名字[(x,y,x,y,probability,name),...]
    """
    result = []
    for box in boxes:
        width, height = image.size
        head_box = (
            width * box[0], height * box[1],
            width * box[2], height * box[3]
        )
        head_image = image.crop(head_box)

        name,trans = resnet18(head_image)
        result.append({
            "box":box[:5],
            "name":name,
            "trans":trans
        })

    app.logger.debug(result)
    return result
