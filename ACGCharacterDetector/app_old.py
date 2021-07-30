import io
import json

from flask import Flask, jsonify
from flask import request
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

app = Flask(__name__)
imagenet_class_index = json.load(open('./static/imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]