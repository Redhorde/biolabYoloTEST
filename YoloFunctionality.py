import cv2
import numpy as np
import os


def yolo_cut(image):
    net = cv2.dnn.readNet("hand-obj_final.weights", "hand-obj.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0]-1]for i in net.getUnconnectedOutLayers()]

    #img = cv2.imread("hand.JPG")
    img = image
    height, width, channels = img.shape
    #cv2.imshow("Hand", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputlayers)
    #print(outputs)

    class_ids = []
    confidences = []
    boxes = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)
                #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
    font = cv2.FONT_HERSHEY_PLAIN
    print("Hand bounding box:")
    print(boxes)
    x = boxes[0][0]
    y = boxes[0][1]
    w = boxes[0][2]
    h = boxes[0][3]
    #print(str(x)+" "+str(y)+" "+str(w)+" "+str(h))
    expand = 50 #expand mask by number of pixels
    img_crop = img[y-expand:y+h+expand, x-expand:x+w+expand]
    # cv2.imshow("Hand_cropped", img_crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("hand_crop.JPG", img_crop)
    return img_crop


def vgg_detect():
    pass


def main():
    for folder in os.scandir("Dataset"):
        for file in os.listdir(folder.path):
            pass


if __name__ == '__main__':
    main()