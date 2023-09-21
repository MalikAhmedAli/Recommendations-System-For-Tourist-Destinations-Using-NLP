import os
from flask import Flask, request, send_file
from predictor import predict as p
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict")
def predict():
    args = request.args
    q = args["q"]
    return p(q)
    return args

@app.route("/image")
def image():
    image_name = request.args["q"]
    image_path = os.path.join("..", "RecommendationSystemCode", "data", "location_images", image_name+".jpg")
    print(image_path)
    try:
        return send_file(image_path, mimetype='image/jpg')
    except Exception as e:
        image_path = os.path.join("..", "RecommendationSystemCode", "data", "location_images", "Swabi.jpg")

        return send_file(image_path, mimetype='image/jpg')

if __name__ == "__main__":
         app.run(host="0.0.0.0")
