import cv2
import matplotlib.pyplot as plt
import numpy as np

cfg_file = '/content/yolov3.cfg'
weight_file = '/content/yolov3.weights'
namesfile = '/content/coco.names'

net = cv2.dnn.readNet(weight_file, cfg_file)

with open(namesfile, 'r') as f:classes = f.read().strip().split('\n')

image_path = '/content/hit.jpg'
image = cv2.imread(image_path)

height, width = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

layer_names = net.getUnconnectedOutLayersNames()

outs = net.forward(layer_names)

class_ids = []
confidences = []
boxes = []

conf_threshold = 0.5

for out in outs:
    for detection in out:scores = detection[5:]
class_id = np.argmax(scores)
confidence = scores[class_id]
if confidence > conf_threshold:center_x = int(detection[0] * width)
center_y = int(detection[1] * height)
w = int(detection[2] * width)
h = int(detection[3] * height)

x = int(center_x - w / 2)
y = int(center_y - h / 2)

class_ids.append(class_id)
confidences.append(float(confidence))
boxes.append([x, y, w, h])

nms_threshold = 0.4
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices.flatten(): x, y, w, h = boxes[i]
label = str(classes[class_ids[i]])
confidence = confidences[i]

cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()