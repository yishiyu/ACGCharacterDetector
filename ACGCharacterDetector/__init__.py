from flask import Flask

app = Flask(__name__)

# 读取配置文件
import configparser
config = configparser.ConfigParser()
config.read("config.ini")

app.config['YOLO_PATH'] = config['YOLO']['yolo_path']
app.config['MODEL_PATH'] = config['YOLO']['model_path']

from .yolo import model as acgmodel


@app.route('/')
def hello():
    print('acgmodel loaded')
    print(acgmodel)
    return 'Hello World!'
