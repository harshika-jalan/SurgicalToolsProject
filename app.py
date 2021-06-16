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
from flask import Flask, request, Response, jsonify, send_from_directory, abort, render_template, render_template_string, make_response, redirect, url_for
import os
import json
import base64
import io
from PIL import Image
from base64 import decodestring
import matplotlib.pyplot as plt
from random import randrange

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

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('file_upload.html')

@app.route('/webcam', methods=['GET','POST'])
def webcam():
    if request.method == 'POST':
        global surgery_name
        surgery_name = request.form.get("surgery_name")

    return render_template_string('''
<video id="video" width="640" height="480" autoplay style="background-color: grey"></video>
<button id="take">Take Photo</button>
<button id="send">Send Photos</button>
<canvas id="canvas" width="640" height="480" style="background-color: grey"></canvas>

<script>

// Elements for taking the snapshot
var video = document.getElementById('video');
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var localstream;
var formdata =  new FormData();

// Get access to the camera!
if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        //video.src = window.URL.createObjectURL(stream);
        video.srcObject = stream;
        localstream = stream;
        video.play();
    });
}

// Send photos
document.getElementById("send").addEventListener("click", function() {
    context.drawImage(video, 0, 0, 640, 480); // copy frame from <video>
    canvas.toBlob(upload, "image/jpeg");  // convert to file and execute function `upload`

});
// Trigger photo take
document.getElementById("take").addEventListener("click", function() {
    context.drawImage(video, 0, 0, 640, 480); // copy frame from <video>
    canvas.toBlob(store, "image/jpeg");  // convert to file and execute function `upload`

});
function store(file) {
    formdata.append("snap", file);
}

function upload(file) {
    // create AJAX requests POST with file
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "{{ url_for('upload') }}", true);
    xhr.onload = function() {
        if(this.status = 200) {
            //console.log(this.response);
            window.location.href = '/result';
        } else {
            console.error(xhr);
        }
        alert(this.response);
    };
    vidOff()
    xhr.send(formdata);

}
function vidOff() {
  //clearInterval(theDrawLoop);
  //ExtensionData.vidStatus = 'off';
  video.pause();
  video.src = "";
  localstream.getTracks()[0].stop();
  console.log("Vid off");
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
        instrument_dict = {}
        for instrument in instrument_list:
            if instrument in instrument_dict:
                instrument_dict[instrument] = instrument_dict[instrument] + 1
            else:
                instrument_dict[instrument] = 1
        surgeries[name] = instrument_dict
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
        if len(add_instruments_string) != 0:#adding instruments
            add_instrument_list = add_instruments_string.split(', ')
            add_instrument_dict = surgeries[name]
            for instrument in add_instrument_list:
                if instrument in add_instrument_dict:
                    add_instrument_dict[instrument] = add_instrument_dict[instrument] + 1
                else:
                    add_instrument_dict[instrument] = 1
            surgeries[name] = add_instrument_dict

        remove_instruments_string = request.form.get("removeInstrument")
        if len(remove_instruments_string) != 0:#removing instruments
            remove_instrument_list = remove_instruments_string.split(', ')
            remove_instrument_dict = surgeries[name]
            for instrument in remove_instrument_list:
                if instrument in remove_instrument_dict:
                    if remove_instrument_dict[instrument]>1:
                        remove_instrument_dict[instrument] = remove_instrument_dict[instrument] - 1
                    else:
                        del remove_instrument_dict[instrument]
            surgeries[name] = remove_instrument_dict

        serialize(surgeries)
        return render_template("success.html")



# API that returns JSON with classes found in images

#@app.route('/upload', methods=['GET','POST'])
def upload2():
    if request.method == 'POST':
        images = request.files.getlist('snap')
        return redirect(url_for('nameform'))

@app.route('/nameform', methods=['GET','POST'])
def nameform():
    return render_template("file_upload.html")

@app.route('/result', methods=['GET','POST'])
def result():
    missing_instruments_string = ""
    wrong_instruments_string = ""
    for key, value in missingInstruments.items():
        missing_instruments_string = '\n'.join([missing_instruments_string, key + " : " + str(value)])
    for key, value in wrongInstruments.items():
        wrong_instruments_string = '\n'.join([wrong_instruments_string, key + " : " + str(value)])
    return render_template("display.html", data=response, name=surgery_name, missing=missing_instruments_string, wrong=wrong_instruments_string, path=detected_classes_image_path)


@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        images = request.files.getlist('snap')
        #global surgery_name
        #surgery_name = "Eye Surgery"
        raw_images = []
        image_names = []
        image_stitching = False
        #surgery_name = request.form.get("surgery_name")
        #images = request.files.getlist("file")
        for image in images:
            image_name = image.filename
            image_names+=[image_name]
            image.save(os.path.join(os.getcwd(), image_name))
            img_raw = tf.image.decode_image(
                open(image_name, 'rb').read(), channels=3)
            raw_images.append(img_raw)
        if len(raw_images)>1:
            raw_images = [stitch(raw_images[0], raw_images[1])]
            image_stitching = True
        num = 0

        # create list for final response
        global response
        global detected_classes_image_path
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
            if image_stitching:
                img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR) #image stitching
            else:
                img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
            detected_classes_image_path = './static/' + 'detection' + str(num) + '.jpg'
            cv2.imwrite('./static/' + 'detection' + str(num) + '.jpg', img)
            print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))
        global missingInstruments
        global wrongInstruments
        missingInstruments, wrongInstruments = check(surgery_name, response)
        print(missingInstruments)
        print(wrongInstruments)
        #remove temporary images
        print("executing")
        for name in image_names:
            os.remove(name)
        #try:
            #return jsonify({"response":response}), 200
            #print("executing2")
            #return render_template("display.html", data=images, name=surgery_name, missing=missingInstruments, wrong=wrongInstruments)

        #except FileNotFoundError:
            #abort(404)

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
    givenInstruments = {}
    for r in response:
        for detection in r['detections']:
            if detection['class'] in givenInstruments:
                givenInstruments[detection['class']] = givenInstruments[detection['class']] + 1
            else:
                givenInstruments[detection['class']] = 1
    missingInstruments = {}#[instrument for instrument in requiredInstruments if instrument not in givenInstruments]
    for instrument, number in requiredInstruments.items():
        if instrument not in givenInstruments:
            missingInstruments[instrument] = number
        elif givenInstruments[instrument] < number:
            missingInstruments[instrument] = number - givenInstruments[instrument]
    wrongInstruments = {} #[instrument for instrument in givenInstruments if instrument not in requiredInstruments]
    for instrument, number in givenInstruments.items():
        if instrument not in requiredInstruments:
            wrongInstruments[instrument] = number
        elif requiredInstruments[instrument] < number:
            wrongInstruments[instrument] = number - requiredInstruments[instrument]
    return missingInstruments, wrongInstruments

#Function for image stitching (2 images)
def stitch(img1, img2) :
    #img1= cv2.imread(os.path.join(os.getcwd(), first))
    #img2 = cv2.imread(os.path.join(os.getcwd(), second))
    img1 = img1.numpy()
    img2 = img2.numpy()
    img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # Create our ORB detector and detect keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=2000)

    # Find the key points and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2,k=2)


    # Finding the best matches
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)
    # Set minimum match condition
    #MIN_MATCH_COUNT = 10

    #if len(good) > MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    result = warpImages(img2, img1, M)

    return result

def warpImages(img1, img2, H):

  rows1, cols1 = img1.shape[:2]
  rows2, cols2 = img2.shape[:2]

  list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
  temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

  # When we have established a homography we need to warp perspective
  # Change field of view
  list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

  list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

  [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

  translation_dist = [-x_min,-y_min]

  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

  output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
  output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

  return output_img

if __name__ == '__main__':
    #app.run(debug=True, host = '0.0.0.0', port=5000)
    #app.run(debug=True, host = '0.0.0.0', port=5000, ssl_context='adhoc')
    app.run(debug=True, port=5000)
