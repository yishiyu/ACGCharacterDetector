import io
from PIL import Image
from flask_cors import CORS
from flask import Flask, jsonify,request,render_template

app = Flask(__name__)
CORS(app)

# 读取配置文件
import configparser
config = configparser.ConfigParser()
config.read("config.ini")

app.config['YOLO_PATH'] = config['YOLO']['yolo_path']
app.config['MODEL_PATH'] = config['YOLO']['model_path']

# 不可把这个调到前面,因为 acgmodel 的初始化需要配置信息
from .yolo import model as acgmodel


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        # 读取传来的图片
        file = request.files['image']
        imagebytes = file.read()
        box = __detect(imagebytes)

        app.logger.debug(box)

        result = {'result':box}
        return jsonify(result)

    return jsonify({'result':[
        # xy, xy
        [0.241746723651886, 0.17238768935203552, 0.4108858108520508, 0.3440185487270355, 0.931640625, 0.0], 
    ]})


def __detect(imagebytes):
    image = Image.open(io.BytesIO(imagebytes))
    box = acgmodel(image).xyxyn[0]

    # app.logger.debug(type(box))
    # app.logger.debug(box)

    return box.tolist()