#from img_text_recognition import*
#from text_recognition_video import*

# USAGE
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_01.jpg
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05

# import the necessary packages

from imutils.video import VideoStream
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import keyboard

from imutils.object_detection import non_max_suppression
import numpy as np
from PIL import Image
import time
import pandas as pd
import imutils
import itertools
import jinja2
import datetime

import pytesseract
import argparse
import cv2
import re
import os

from flask import Flask, render_template, request, redirect, session,url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, logout_user


app = Flask(__name__)
UPLOAD_FOLDER = r"C:\Users\hp\Desktop\ml\static"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///marathonreg.db'
db = SQLAlchemy(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Initialize login manager
login = LoginManager(app)
login.init_app(app)


#_________________________________________________________Login_________________________________________________________
@login.user_loader
def load_user(id):
    return User.query.get(int(id))

class User(db.Model):
    """ Create user table"""
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(40))
    lname = db.Column(db.String(40))
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(80))
    phone = db.Column(db.String(80))
    password = db.Column(db.String(80))

    def __init__(self, fname, lname, username , email , phone , password):
        self.fname = fname
        self.lname = lname
        self.username = username
        self.email = email
        self.phone = phone
        self.password = password



@app.route('/signup', methods=['GET', 'POST'])
def register():

    if request.method == "POST":
        fname = request.form['fname']
        lname = request.form['lname']
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']

        register = User(fname=fname, lname=lname, username=username, email=email, phone=phone, password=password)
        db.session.add(register)
        db.session.commit()

        if register is None:
            return render_template("signup.html")

        return redirect(url_for("login"))
    return render_template("signup.html")



@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        login = User.query.filter_by(username=username, password=password).first()
        if login is not None:
            return redirect(url_for("imgupload"))
            #return render_template("upload.html")
    return render_template("Login.html")


#***********************************************************************************************************************
#***************************************************Img Upload**********************************************************

@app.route('/imgupload')
def img():
    return render_template('imgupload.html')

@app.route('/imgupload', methods=['POST'])
def imgupload():
    '''
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        #f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        #userimg = f.filename
        k = imgreg(filename)
        PEOPLE_FOLDER = os.path.join('static', 'results')
        UPLOAD_FOLDER2 = PEOPLE_FOLDER
        app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2
        full_filename = os.path.join(app.config['UPLOAD_FOLDER2'], k[1])
        os.chdir(r"C:\Desktop\ml\images")
            #return render_template("complete.html", image_name=filename)
    return render_template("success_img.html", name=file.filename,value=k[0], user=full_filename)
'''
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))

        userimg = f.filename
        k = imgreg(userimg)
        PEOPLE_FOLDER = os.path.join('static', 'result_image')
        UPLOAD_FOLDER2 = PEOPLE_FOLDER
        app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2
        full_filename = os.path.join(app.config['UPLOAD_FOLDER2'], k[1])
        os.chdir(r"C:\Users\hp\Desktop\ml")
    return render_template("success_img.html", name=f.filename, value=k[0], user=full_filename)

def decode_predictions(scores, geometry, args_i):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < args_i["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


def filterText(text):
    text = "".join([c if 48 <= ord(c) <= 57 or 65 <= ord(c) <= 90 else "" for c in text]).strip()
    regex = '^[HFM]\d\d\d\d$'
    # regex = '^\d\d\d\d$'
    #regex = '[\d]+'
    print(text)
    if re.search(regex, text):
        print("Predicted Text")
        print("========")
        print("{} \n".format(text))

        return text
    else:
        return None


def imgreg(userimg):
        args_i = {
            #"image": "images/"+filename,
            "image": userimg,
            "east": "frozen_east_text_detection.pb",
            "min_confidence": 0.5,
            "width": 320,
            "height": 320,
            "padding": 0.01,
        }

        # load the input image and grab the image dimensions
        #image = cv2.imread(args_i["image"])
        image = cv2.imread("C:\\Users\\hp\\Desktop\\ml\\static\\" + args_i["image"])
        orig = image.copy()
        (origH, origW) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (args_i["width"], args_i["height"])
        rW = origW / float(newW)
        rH = origH / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(args_i["east"])

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry, args_i)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # initialize the list of results
        results = []
        predictedTexts=[]

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * args_i["padding"])
            dY = int((endY - startY) * args_i["padding"])

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))

            # extract the actual padded ROI
            roi = orig[startY:endY, startX:endX]

            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 4, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text
            config = ("-l eng --oem 1 --psm 7")
            text = filterText(pytesseract.image_to_string(roi, config=config))
            if text:
                # add the bounding box coordinates and text
                # print("yes")
                results.append(((startX, startY, endX, endY), text))
                predictedTexts.append(text)

        # sort the results bounding box coordinates from top to bottom
        results = sorted(results, key=lambda r: r[0][1])
        output = orig.copy()
        #cv2.imshow("Text Detection", output)
        # loop over the results
        for ((startX, startY, endX, endY), text) in results:
            # display the text OCR'd by Tesseract
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw the text and a bounding box surrounding
            # the text region of the input image

            cv2.rectangle(output, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(output, text, (startX, startY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        os.chdir(r"C:\Users\hp\Desktop\ml\static\result_image")
        rnd = time.time()
        fname = str(rnd) + ".jpg"
        cv2.imwrite(fname, output)
        os.chdir(r"C:\Users\hp\Desktop\ml")

        return predictedTexts, fname

#***********************************************************************************************************************

#_______________________________________________Video/webcam__________________________________________

env = jinja2.Environment()
env.globals.update(zip=zip)

def decode_predictions_video(scores, geometry, args):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):

        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):

            if scoresData[x] < args["min_confidence"]:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)


def filterText_video(text):
    text = "".join([c if 48 <= ord(c) <= 57 or 65 <= ord(c) <= 90 else "" for c in text]).strip()
    #regex = '^[HFM]\d\d\d\d$'
    regex = '^\d\d\d\d$'

    if re.search(regex, text):
        return text
    else:
        return None


def printTime(millis):
    millis = int(millis)
    seconds = int((millis / 1000) % 60)
    seconds = int(seconds)
    minutes = int((millis / (1000 * 60)) % 60)
    minutes = int(minutes)
    hours = int((millis / (1000 * 60 * 60)) % 24)

    print("%d:%d:%d" % (hours, minutes, seconds))
    return str(hours) + ":" + str(minutes) + ":" + str(seconds)



#***************************************************Video Upload********************************************************

@app.route('/videoupload')
def video():
    return render_template('videoupload.html')

@app.route('/videoupload', methods=['POST'])
def videoupload():
    '''
    target = os.path.join(APP_ROOT,'videos/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)
    '''
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        e = request.files['excel']
        e.save(os.path.join(app.config['UPLOAD_FOLDER'], e.filename))
        PEOPLE_FOLDER = os.path.join('static', 'excel')
        UPLOAD_FOLDER2 = PEOPLE_FOLDER
        app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2

        uservideo = f.filename

        userexcel = e.filename
        k = videoreg(uservideo, userexcel)

    return render_template("success_video.html", name=f.filename, value=k[0], user=k[1], m=zip(k[0], k[1], k[2]),T=k[2], excel=e.filename)



def videoreg(uservideo,userexcel):
    args = {
        "east": "frozen_east_text_detection.pb",
        "min_confidence": 0.5,
        "width": 320,
        "height": 320,
        "padding": 0.03,
        "video": uservideo,  # path of video
        "excel": userexcel,
    }

    (W, H) = (None, None)  # actual dimensions of image 720x460 etc
    (newW, newH) = (args["width"], args["height"])  # required dimewnsion
    (rW, rH) = (None, None)  # ration of both

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",  # Scores - probability
        "feature_fusion/concat_3"]  # geometry - dimensions of the bounding box

    print("loading EAST text detector...")

    net = cv2.dnn.readNet(args["east"])

    # if no video path, grabbing the reference to the web cam
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = WebcamVideoStream(src=0).start()  # 0 for default webcam
        # time.sleep(1.0)

    else:
        vs = cv2.VideoCapture("C:\\Users\\hp\\Desktop\\ml\\static\\" + args["video"])

    fps = FPS().start()

    fnumber = -10
    predictedTexts = []

    vfname = []
    T = []
    # loop over frames from the video stream
    while True:
        if ("C:\\Users\\hp\\Desktop\\ml\\static\\" + args["video"]):
            fnumber += 10
            vs.set(cv2.CAP_PROP_POS_FRAMES, fnumber)

        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame

        # check to see if we have reached the end of the stream
        if frame is None:
            break
        if keyboard.is_pressed('q'):
            break
        # resize the frame maintain aspect ratio
        frame = imutils.resize(frame, width=1000)
        orig = frame.copy()
        # cv2.imshow(frame)
        if W is None or H is None:
            (H, W) = frame.shape[:2]  # actual size
            rW = W / float(newW)
            rH = H / float(newH)

        # resize the frame
        frame = cv2.resize(frame, (newW, newH))

        # construct a blob
        blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)

        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # decode the predictions obtaining probabilites and position of box
        (rects, confidences) = decode_predictions_video(scores, geometry, args)

        boxes = non_max_suppression(np.array(rects), probs=confidences)

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        for (startX, startY, endX, endY) in boxes:
            # scaling the bounding box coordinates based on the respective ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # applying padding in percentage
            dX = int((endX - startX) * args["padding"])
            dY = int((endY - startY) * args["padding"])

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(W, endX + (dX * 2))
            endY = min(H, endY + (dY * 2))

            # extract the actual padded image out of original
            roi = orig[startY:endY, startX:endX]

            # config stating language, LSTM model and stating all is one line of text
            config = ("-l eng --oem 1 --psm 7")

            # obtaining text out of image
            text = filterText_video(pytesseract.image_to_string(roi, config=config))

            if text and text not in predictedTexts:
                # add the bounding box coordinates and text

                print("Predicted Text")
                print("========")

                # timestamps for webcam wil be in realtime whereas for video will be according to video
                if args["video"]:
                    print(text, " at time ~ ", end="")
                    T.append(printTime(vs.get(cv2.CAP_PROP_POS_MSEC)))  # converting millisecs into hour min secs
                else:
                    print("{} at {} \n".format(text, datetime.datetime.now().strftime("%H:%M:%S on %d/%m/%Y")))

                predictedTexts.append(text)
                # draw the bounding box on the frame
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

                fps.update()

                # show the output frame
                # cv2.imshow("Text Detection", orig)

                # if the `q` key was pressed, break from the loop
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord("q"):
                # break
                if keyboard.is_pressed('q'):
                    break


                os.chdir(r"C:\Users\hp\Desktop\ml\static")
                rnd = time.time()
                fname = str(rnd) + ".jpg"
                cv2.imwrite(fname, orig)
                os.chdir(r"C:\Users\hp\Desktop\ml")
                vfname.append(fname)
        if keyboard.is_pressed('q'):
            break

    # stop the timer and display FPS information
    fps.stop()
  
    print(vfname)
    # if we are using a webcam, release the pointer
    if not args["video"]:
        vs.stop()
    # otherwise, release the file pointer of video
    else:
        vs.release()

    # close all windows
    cv2.destroyAllWindows()


    data = pd.read_csv("C:\\Users\\hp\\Desktop\\ml\\static\\" + args["excel"])
    df = pd.DataFrame(data, columns=['Bib_no'])

    bib_list = df.values.tolist()
    print('bib_list : ', bib_list)

    print('predictedTexts : ', predictedTexts)
    n = len(bib_list)

    pred = []
    for el in predictedTexts:
        sub = el.split(', ')
        pred.append(sub)

    print('pred:', pred)

    if (len(pred) != n):
        pred.append(None)
    print('pred:', pred)

    p = len(pred)

    b = []
    for i in range(n):
        for j in range(p):
            if (bib_list[i] == pred[j]):
                df.loc[i, 'Status'] = T[j]
                # df.loc[i, 'Time and Date'] = T[j]
                break
            else:
                df.loc[i, 'Status'] = 'Not Predicted'

            # df.Status.fillna("Not predicted", inplace=True)


    print(df)

    download_source = (r'C:\Users\hp\Desktop\ml\static\Vexcel\output_video.xlsx')
    df.to_excel(download_source)

    return predictedTexts, vfname, T


#***********************************************************************************************************************
#**************************************************Webcam***************************************************************

@app.route('/webcam')
def webc():
    return render_template('webcam.html')

@app.route('/webcam', methods=['POST'])
def webcam():
    if request.method == 'POST':
        e = request.files['excel']
        e.save(os.path.join(app.config['UPLOAD_FOLDER'], e.filename))


    userexcel = e.filename

    k = webreg( userexcel)
    return render_template("success_webcam.html", m=zip(k[0], k[1], k[2]),excel=e.filename)


def webreg(userexcel):
    args = {
        "east": "frozen_east_text_detection.pb",
        "min_confidence": 0.5,
        "width": 320,
        "height": 320,
        "padding": 0.0,
        "webcam": "",  # path of webcam
        "excel": userexcel,
    }

    (W, H) = (None, None)  # actual dimensions of image 720x460 etc
    (newW, newH) = (args["width"], args["height"])  # required dimewnsion
    (rW, rH) = (None, None)  # ration of both

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",  # Scores - probability
        "feature_fusion/concat_3"]  # geometry - dimensions of the bounding box

    print("loading EAST text detector...")

    net = cv2.dnn.readNet(args["east"])

    # if no webcam path, grabbing the reference to the web cam

    print("[INFO] starting webcam stream...")
    vs = WebcamVideoStream(src=0).start()  # 0 for default webcam
    # time.sleep(1.0)

    fnumber = -10
    vfname = []
    T = []
    predictedTexts = []
    fps = FPS().start()
    while True:
        if args["webcam"]:
            fnumber += 10
            vs.set(cv2.CAP_PROP_POS_FRAMES, fnumber)

        frame = vs.read()
        frame = frame[1] if args.get("webcam", False) else frame

        # check to see if we have reached the end of the stream
        if frame is None:
            break

        # resize the frame maintain aspect ratio
        frame = imutils.resize(frame, width=1000)
        orig = frame.copy()

        if W is None or H is None:
            (H, W) = frame.shape[:2]  # actual size
            rW = W / float(newW)
            rH = H / float(newH)

        # resize the frame
        frame = cv2.resize(frame, (newW, newH))

        # construct a blob
        blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)

        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # decode the predictions obtaining probabilites and position of box
        (rects, confidences) = decode_predictions_video(scores, geometry, args)

        boxes = non_max_suppression(np.array(rects), probs=confidences)

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        for (startX, startY, endX, endY) in boxes:
            # scaling the bounding box coordinates based on the respective ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # applying padding in percentage
            dX = int((endX - startX) * args["padding"])
            dY = int((endY - startY) * args["padding"])

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(W, endX + (dX * 2))
            endY = min(H, endY + (dY * 2))

            # extract the actual padded image out of original
            roi = orig[startY:endY, startX:endX]

            # config stating language, LSTM model and stating all is one line of text
            config = ("-l eng --oem 1 --psm 7")

            # obtaining text out of image
            text = filterText_video(pytesseract.image_to_string(roi, config=config))
            # text = pytesseract.image_to_string(roi, config=config)
            if text and text not in predictedTexts:
                # add the bounding box coordinates and text

                print("Predicted Text")
                print("========")

                # timestamps for webcam wil be in realtime whereas for webcam will be according to webcam
                if args["webcam"]:
                    print(text, " at time ~ ", end="")
                    printTime(vs.get(cv2.CAP_PROP_POS_MSEC))  # converting millisecs into hour min secs
                else:
                    Tt = datetime.datetime.now().strftime("%H:%M:%S on %d/%m/%Y")
                T.append(Tt)
                predictedTexts.append(text)
                # draw the bounding box on the frame
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

                os.chdir(r"C:\Users\hp\Desktop\ml\static")
                rnd = time.time()
                fname = str(rnd) + ".jpg"
                cv2.imwrite(fname, orig)
                os.chdir(r"C:\Users\hp\Desktop\ml")
                vfname.append(fname)

        fps.update()

        # show the output frame
        cv2.imshow("Text Detection", orig)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # if we are using a webcam, release the pointer
    if not args["webcam"]:
        vs.stop()
    # otherwise, release the file pointer of webcam
    else:
        vs.release()

    # close all windows
    cv2.destroyAllWindows()

    data = pd.read_csv("C:\\Users\\hp\\Desktop\\ml\\static\\" + args["excel"])
    df = pd.DataFrame(data, columns=['Bib_no'])

    bib_list = df.values.tolist()
    print('bib_list : ', bib_list)

    print('predictedTexts : ', predictedTexts)
    n = len(bib_list)

    pred = []
    for el in predictedTexts:
        sub = el.split(', ')
        pred.append(sub)

    print('pred:', pred)

    if (len(pred) != n):
        pred.append(None)
    print('pred:', pred)

    p = len(pred)

    b = []
    for i in range(n):
        for j in range(p):
            if (bib_list[i] == pred[j]):
                df.loc[i, 'Status'] = T[j]
                #df.loc[i, 'Time and Date'] = T[j]
                break
            else:
                df.loc[i, 'Status'] = 'Not Predicted'

    #df.Status.fillna("Not predicted", inplace=True)

    print(df)

    download_source = (r'C:\Users\hp\Desktop\ml\static\Wexcel\output_video.xlsx')
    df.to_excel(download_source)

    return predictedTexts, vfname, T


#________________________________________________Logout_________________________________________________________________

@app.route("/logout", methods=['GET'])
def logout():

    # Logout user
    logout_user()
    #flash('You have logged out successfully', 'success')
    return redirect(url_for("login"))



if __name__ == '__main__':
    app.debug = True
    db.create_all()
    app.secret_key = "123"
    app.run()
