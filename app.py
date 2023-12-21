import base64
import numpy as np
import io
from PIL import Image
from flask import Flask, render_template, redirect, request, jsonify
import tensorflow as tf
from tensorflow import keras
from model import ImageGenModel
from postprocess import PostProcess
import time
model_path = "g_model_052000.h5"
post_model_path = "model_10.h5"

main_model = ImageGenModel(model_path)
post_model = PostProcess(post_model_path)

app = Flask(__name__)


@app.route("/predicted_image", methods=["POST"])
def generated():
    if "drawing" not in request.json:
        return "No drawing data received", 400

    drawing_file = request.json["drawing"]

    if drawing_file:
        drawing_image = Image.open(
            io.BytesIO(base64.b64decode(drawing_file.split(",")[1]))
        )
        img = drawing_image.convert("RGBA")

        new_img = Image.new("RGBA", img.size, (255, 255, 255))

        # Paste the original image onto the new image
        new_img.paste(img, (0, 0), img)

        target_size = (256, 256)  # Set your desired target size here
        input_img = new_img.resize(target_size)

        input_img = input_img.convert("RGB")


        start_time = time.time()  # Record the start time

        img = main_model.generate(input_img)
        img_gen = img[0]

        img_post = post_model.process(input_img, img_gen)
        img_post_advanced = post_model.process(
            input_img,
            img_gen,
            model_cycles=1,
            advanced=True,
            border_padding=2,
        )

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        res = Image.fromarray(np.uint8(img_post_advanced[0]))

        image_io = io.BytesIO()
        res.save(image_io, format="JPEG")

        image_io.seek(0)

        image_url = "data:image/jpeg;base64," + base64.b64encode(
            image_io.read()
        ).decode("utf-8")

        response_data = {"image": image_url, "time": elapsed_time}

        return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0',port=8080)
