import cv2


def show_object_box(box,img):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), [0, 255, 0], 2)
    cv2.imshow("test", img)
    cv2.waitKey(1)