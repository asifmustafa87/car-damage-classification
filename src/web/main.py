# # all imports
import os
from pathlib import Path

import matplotlib.pyplot as plt
import logging
import cv2
import numpy as np
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    url_for,
    send_from_directory,
    send_file,
)

from src.ml.handle_prediction import (
    PreProcessedImage,
    predict_with_explain,
    correct_model_on_images,
)

from werkzeug.utils import secure_filename
from keras.models import load_model
import json
import csv
import math

UPLOAD_FOLDER = "static/uploads/"
CURRENT_DIRECTORY = Path(__file__).parent
SRC_DIRECTORY = CURRENT_DIRECTORY.parent
os.chdir(CURRENT_DIRECTORY)

app = Flask(__name__)

app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["IMAGES"] = "images"
app.config["LABELS"] = []
app.config["HEAD"] = 0
app.config["OUT"] = "out.csv"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

inverse_classes = {
    0: "Dent",
    1: "Other",
    2: "Rim",
    3: "Scratch",
}

prediction_class = {v: k for k, v in inverse_classes.items()}

logger = logging.getLogger(__name__)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home_page():
    return render_template("something.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/labeller")
def upload_form():
    return render_template("label.html")


@app.route("/labeller", methods=["POST"])
def upload_image():
    if "files[]" not in request.files:
        flash("No file part")
        return redirect(request.url)
    files = request.files.getlist("files[]")
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    return render_template("label.html", filenames=file_names)


@app.route("/damage", methods=["GET", "POST"])
def main():
    return render_template("damage_detection.html")


@app.route("/label", methods=["GET", "POST"])
def basiclabel():
    return render_template("Basic_Label.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


def model_predict(img_path):
    logger.info("Model processing image: %s", img_path.split("/")[-1])
    image = cv2.imread(img_path)
    image = PreProcessedImage(image)
    predicted_label, softmax_output, heatmap_grid = predict_with_explain(image)
    predicted_confidence = "{:.1f}".format(max(softmax_output) * 100)

    result = f"Prediction of the model is {inverse_classes[predicted_label].lower()}"
    return result, predicted_confidence


@app.route("/submit", methods=["GET", "POST"])
def get_output():
    if request.method == "POST":
        if "form1" in request.form:
            files = request.files.getlist("files[]")
            file_names = []
            name = []
            p = []
            perc = []
            error = 0
            if len(files) > 50:
                error = "You are only allowed to upload a maximum of 50 files"
                return render_template("damage_detection.html", error=error)
            else:
                for img in files:
                    if allowed_file(img.filename):
                        img_path = "static/" + img.filename
                        img.save(img_path)
                        i, j = model_predict(img_path)
                        p.append(i)
                        perc.append(j)
                        file_names.append(img_path)
                        name.append(img.filename)
                    else:
                        error = "Wrong filetype"
                return render_template(
                    "damage_detection.html",
                    prediction=p,
                    img_path=file_names,
                    leng=len(file_names),
                    img_name=name,
                    url="static/plot.png",
                    perc=perc,
                    error=error,
                )
        else:

            your_structure = request.form.to_dict()
            del your_structure["form2"]
            print(your_structure)
            num = 0
            temp1 = []
            with open("static/corrections/corrections.csv", "w") as file:
                writer = csv.writer(file)
                label_list = ["Label", "File_name"]
                writer.writerow(label_list)
            for key in your_structure:
                if key[0:5] == "Damag" and num == 0:
                    temp1.append(your_structure[key])
                    num = 1
                elif key[0:5] == "Image" and num == 1:
                    temp1.append(your_structure[key])
                    with open(
                        "static/corrections/corrections.csv", "a", newline=""
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(temp1)
                    num = 0
                    img_path = "static/" + your_structure[key]
                    cor = temp1[0]

                    correction_label = prediction_class[cor.capitalize()]
                    image = cv2.imread(img_path)
                    image = PreProcessedImage(image)
                    correct_model_on_images(image, correction_label)

                    temp1 = []

            return render_template("download.html")


@app.route("/basic", methods=["GET", "POST"])
def label():
    if request.method == "POST":
        if "form1" in request.form:
            files = request.files.getlist("files[]")
            file_names = []
            name = []
            error = 0
            if len(files) > 200:
                error = "You are only allowed to upload a maximum of 200 files"
                return render_template("Basic_Label.html", error=error)
            else:
                for img in files:
                    if allowed_file(img.filename):
                        img_path = "static/" + img.filename
                        img.save(img_path)
                        file_names.append(img_path)
                        name.append(img.filename)
                    else:
                        error = "Wrong filetype"
                return render_template(
                    "Basic_Label.html",
                    img_path=file_names,
                    leng=len(file_names),
                    img_name=name,
                    error=error,
                )
        else:
            your_structure = request.form.to_dict()
            del your_structure["form2"]
            temp1 = {}
            temp2 = []
            num = 1
            for key in your_structure:
                if key[0:5] == "Image" and num == 1:
                    temp1["file_name"] = your_structure[key]
                    num = 0
                elif key[0:5] == "Damag" and num == 0:
                    temp1["Label"] = your_structure[key]
                    temp2.append(temp1)
                    num = 1
                    temp1 = {}
            temp3 = {"annotations": temp2}
            print(temp2)
            with open("annotations.json", "w") as f:
                json.dump(temp3, f)
            return render_template("downloader.html")


@app.route("/test", methods=["GET"])
def return_file():
    return send_file("static/corrections/corrections.csv", as_attachment=True)


@app.route("/log", methods=["GET"])
def log():
    return send_file("../../logs/retraining.log", as_attachment=True)


@app.route("/downloader")
def downloder():
    return send_file(
        "annotations.json", mimetype="text/json", as_attachment=True
    )


# @app.route('/results', methods=['GET', 'POST'])
# def results():
#   if request.method == 'POST':
#      print(filename)
#     with open('file.json', 'w') as f:
#        json.dump(request.form, f)
# return render_template('result.html')


with open("out.csv", "w") as f:
    f.write("image,id,name,xMin,xMax,yMin,yMax\n")


# if __name__ == "__main__":
def execute():
    app.run(host="0.0.0.0", port=8888, debug=False)
