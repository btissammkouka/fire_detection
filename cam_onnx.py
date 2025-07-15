import cv2
import numpy as np
import onnxruntime

# Charger le modèle ONNX
model_path = "best_int8.onnx"
session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Paramètres YOLO
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
conf_threshold = 0.4
nms_threshold = 0.5


# Prétraitement
def preprocess(frame):
    img_resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    img_input = img_resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)  # BGR->RGB, HWC->CHW
    img_input /= 255.0
    img_input = np.expand_dims(img_input, axis=0)
    return img_input


# Post-traitement
def postprocess(outputs, frame):
    predictions = outputs[0]
    predictions = predictions[0]  # Enlever le batch

    boxes = []
    scores = []

    for pred in predictions:
        if len(pred) == 6:
            x_center, y_center, width, height, objectness, class_conf = pred
        else:
            x_center, y_center, width, height = pred[0:4]
            objectness = pred[4]
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]

        conf = objectness * class_conf
        if conf < conf_threshold:
            continue

        x1 = int((x_center - width / 2) * frame.shape[1] / INPUT_WIDTH)
        y1 = int((y_center - height / 2) * frame.shape[0] / INPUT_HEIGHT)
        x2 = int((x_center + width / 2) * frame.shape[1] / INPUT_WIDTH)
        y2 = int((y_center + height / 2) * frame.shape[0] / INPUT_HEIGHT)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, w, h = boxes[i]
        conf = scores[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"Fire {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


# Capture webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)

    frame = postprocess(outputs, frame)

    cv2.imshow("Fire Detection (YOLOv5 ONNX INT8)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
