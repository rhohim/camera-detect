# from flask import Flask, redirect, request, jsonify, render_template, url_for, session, Response
# import cv2

# app=Flask(__name__)
# camera = cv2.VideoCapture(0)

# def gen():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             ret,buffer = cv2.imencode('.jpg',frame)
#             frame = buffer.tobytes()

#         yield(b'--frame\r\n'b'Content-Type:  image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/') 
# def index():
#     return render_template('index.html')

# @app.route('/video')
# def video(): 
#     return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True)


# import the opencv library
from flask import Flask, redirect, request, jsonify, render_template, url_for, session, Response
# from tkinter import Frame
# import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np

app=Flask(__name__)
Wcam, Hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, Wcam)
cap.set(4, Hcam)
cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_and_predict_mask(frame, faceNet, maskNet):
	h = 480
	w = 640
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:

		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

prototxtPath = r"face_detector\\deploy.prototxt"
weightsPath = r"face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")

def gen():
    while True:
        # ret, frame = cap.read()

        # cv2.imshow('test',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        success, img = cap.read()
        (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (box, pred) in zip(locs, preds):

            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Keren lo !" if mask > withoutMask else "Pake Masker ajg !!"
            color = (0, 255, 0) if label == "Keren lo!" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(img, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        if not success:
            break
        else:
            ret,buffer = cv2.imencode('.jpg',img)
            frame = buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type:  image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video(): 
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

# cap.release()
# cv2.destroyAllWindows()
