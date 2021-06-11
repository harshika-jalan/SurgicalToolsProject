import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort, render_template, render_template_string
import os
import json
import base64
import io
from PIL import Image
from base64 import decodestring

#@source https://github.com/theAIGuysCode/Object-Detection-API
# customize your API through the following parameters
classes_path = './data/labels/custom.names'
weights_path = './weights/yolov3.tf'
tiny = False                    # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
output_path = './detections/'   # path to output folder where images with detections are saved
num_classes = 6                # number of classes in model

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')

#method for serializing surgical data to json file
def serialize(surgeries):
    with open("surgical_data.json", "w") as write_file:
        json.dump(surgeries, write_file, indent=4)

#method for deserializing surgical data to json file
def deserialize():
    f = open("surgical_data.json",)
    #returns json object as dictionary
    surgeries=json.load(f)
    #closing file
    f.close()
    return surgeries

surgeries = deserialize()



# Initialize Flask application
app = Flask(__name__)

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        return jsonify(request.form['userID'], request.form['file'])
    return render_template('signup.html')

@app.route('/')
def home():
    #return render_template("file_upload.html")
    #return render_template("file_upload.html")
    return render_template_string('''
<video id="video" width="640" height="480" autoplay style="background-color: grey"></video>
<button id="send">Take & Send Photo</button>
<canvas id="canvas" width="640" height="480" style="background-color: grey"></canvas>

<script>

// Elements for taking the snapshot
var video = document.getElementById('video');
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');

// Get access to the camera!
if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        //video.src = window.URL.createObjectURL(stream);
        video.srcObject = stream;
        video.play();
    });
}

// Trigger photo take
document.getElementById("send").addEventListener("click", function() {
    context.drawImage(video, 0, 0, 640, 480); // copy frame from <video>
    canvas.toBlob(upload, "image/jpeg");  // convert to file and execute function `upload`

});

function upload(file) {
    // create form and append file
    var formdata =  new FormData();
    formdata.append("snap", file);

    // create AJAX requests POST with file
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "{{ url_for('upload') }}", true);
    xhr.onload = function() {
        if(this.status = 200) {
            console.log(this.response);
        } else {
            console.error(xhr);
        }
        alert(this.response);
    };
    xhr.send(formdata);

}



</script>
''')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        #Save the file in images
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'data', 'images', f.filename)
        f.save(file_path)
        return render_template("success.html", name = f.filename)


#APIs to maintain and change surgical data records
#displaying menu
@app.route('/records')
def record():
    return render_template("recordsMenu.html")

#Adding New Surgery
@app.route('/newSurgery')
def newSurgery():
    return render_template("newSurgeryForm.html")

@app.route('/addNewSurgery', methods=['GET', 'POST'])
def addNewSurgery():
    if request.method == "POST":
        name = request.form.get("name")
        instruments_string = request.form.get("instrument")
        instrument_list = instruments_string.split(', ')
        surgeries[name] = instrument_list
        serialize(surgeries)
        return render_template("success.html")
#Adding/Removing instruments from existing surgical procedure
@app.route('/changeInstruments')
def changeInstruments():
    return render_template("changeInstrumentsForm.html")

@app.route('/AddRemoveInstruments', methods=['GET', 'POST'])
def AddRemoveInstruments():
    if request.method == "POST":
        name = request.form.get("name")
        add_instruments_string = request.form.get("addInstrument")
        if len(add_instruments_string) != 0:
            add_instrument_list = add_instruments_string.split(', ')
            surgeries[name] = surgeries[name] + add_instrument_list
        remove_instruments_string = request.form.get("removeInstrument")
        if len(remove_instruments_string) != 0:
            remove_instrument_list = remove_instruments_string.split(', ')
        surgeries[name] = [s for s in surgeries[name] if s not in remove_instrument_list]
        serialize(surgeries)
        return render_template("success.html")

@app.route('/video_feed')
def video_feed(video):
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def gen(video):

    success, image = video.read()
    ret, jpeg = cv2.imencode('.jpg', image)
    frame = jpeg.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# API that returns JSON with classes found in images
@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        images = request.files.getlist('snap')
        surgery_name = "Eye Surgery"
        raw_images = []
        image_names = []
        #surgery_name = request.form.get("surgery_name")
        #images = request.files.getlist("file")
        for image in images:
            image_name = image.filename
            image_names+=[image_name]
            image.save(os.path.join(os.getcwd(), image_name))
            img_raw = tf.image.decode_image(
                open(image_name, 'rb').read(), channels=3)
            raw_images.append(img_raw)

        num = 0

        # create list for final response
        response = []
        detected_classes_image_path = ""

        for j in range(len(raw_images)):
            # create list of responses for current image
            responses = []
            raw_img = raw_images[j]
            num+=1
            img = tf.expand_dims(raw_img, 0)
            img = transform_images(img, size)

            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            print('time: {}'.format(t2 - t1))

            print('detections:')
            for i in range(nums[0]):
                print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))
                responses.append({
                    "class": class_names[int(classes[0][i])],
                    "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100))
                })
            response.append({
                "image": image_names[j],
                "detections": responses
            })
            img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
            detected_classes_image_path = '../detections/' + 'detection' + str(num) + '.jpg'
            print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))
        missingInstruments, wrongInstruments = check(surgery_name, response)
        #remove temporary images
        print("executing")
        for name in image_names:
            os.remove(name)
            try:
                #return jsonify({"response":response}), 200
                print("executing2")
                return render_template("display.html", name=surgery_name, missing=missingInstruments, wrong=wrongInstruments, detected_classes_image=detected_classes_image_path)

            except FileNotFoundError:
                abort(404)

# API that returns image with detections on it
@app.route('/image', methods= ['POST'])
def get_image():
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + 'detection.jpg', img)
    print('output saved to: {}'.format(output_path + 'detection.jpg'))

    # prepare image for response
    _, img_encoded = cv2.imencode('.png', img)
    response = img_encoded.tostring()

    #remove temporary image
    os.remove(image_name)

    try:
        return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)

#method to check which tools are missing and incorrect
def check(name, response):
    requiredInstruments = surgeries[name]
    givenInstruments = []
    for r in response:
        for detection in r['detections']:
            givenInstruments += [detection['class']]
    missingInstruments = [instrument for instrument in requiredInstruments if instrument not in givenInstruments]
    wrongInstruments = [instrument for instrument in givenInstruments if instrument not in requiredInstruments]
    return missingInstruments, wrongInstruments


if __name__ == '__main__':
    #app.run(debug=True, host = '0.0.0.0', port=5000)
    #app.run(debug=True, host = '0.0.0.0', port=5000, ssl_context='adhoc')
    app.run(debug=True, port=5000)
