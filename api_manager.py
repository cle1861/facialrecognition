from flask import Flask, request, jsonify, send_file
import cv2
import pickle
import face_recognition
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "input_images"

# Load up the data to be ready for requests
print("[INFO] loading encodings...")
data = pickle.loads(open("out/out", "rb").read())
image = cv2.imread("input_images/test_bridge.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("[INFO] locating faces...")
boxes = face_recognition.face_locations(rgb, model="cnn")


@app.route("/test_image_fresh", methods=['GET'])
def raw_image_fetch():
    return send_file("input_images/test_bridge.jpg")


@app.route("/test_image", methods=['GET'])
def test_image_fetch():
    return send_file("out/done.jpg")


@app.route("/test", methods=['GET'])
def sfs_test_demo():
    s = time.perf_counter()
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    for encoding in encodings:
        name = "Unknown"
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        if name == "Captain Kirk":
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
            cv2.imwrite("out/done.jpg", image)
            d = time.perf_counter()
            return jsonify((top, right, bottom, left, d-s))
    # return jsonify({"test":1})


if __name__ == '__main__':
    app.run()
