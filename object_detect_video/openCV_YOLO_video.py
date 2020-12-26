import cv2 as cv
import numpy as np
import time
import warnings
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

warnings.filterwarnings(action='ignore')

file_name = "./House Tour_ An eclectic black and white bungalow in Singapore.mp4"

min_confidence = 0.5
frame_width = 800

output_name = "./video/House Tour.avi"


net = cv.dnn.readNet("./PYQT5/yolov3.weights", "./PYQT5/yolov3.cfg")
classes = []
with open("./PYQT5/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv.VideoCapture()

def selectFile():
    global cap

    file_name = filedialog.askopenfilename(initialdir='./video', title='Select File', filetypes=(('MP4 files','*.mp4'), ('all files','*.*')))
    print('file_name :', file_name)
    cap = cv.VideoCapture(file_name)

    detectAndDisplay()

def quitVideo():
    if cap.isOpened():
        cap.release()
        cv.destroyAllWindows()
        sys.exit()
    else:
        pass


def detectAndDisplay():
    _, frame = cap.read()  # 한개의 frame을 불러온다.
    start_time = time.time()
    height, width, channels = frame.shape
    frameSize = int(sizeSpin.get())
    ratio = frameSize / width
    dimension = (frameSize, int(height * ratio))
    frame = cv.resize(frame, dimension, interpolation=cv.INTER_AREA)
    print(height, width, channels)
    height, width, channels = frame.shape

    min_confidence = float(sizeSpin2.get())

    blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    print(len(outs))

    class_ids = []
    confidences = []
    boxes = []

    print(frame_width, min_confidence)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]],confidences[i]*100)
            print(i, label)
            color = colors[i]
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(frame, label, (x, y + 30), font, 0.5, (0, 255, 0), 1)
    end_time = time.time()
    process_time = end_time - start_time
    print("한개의 프레임 처리 시간: ", process_time)
    # cv.imshow("yolo detect frame", frame)
    cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # numpy배열을 image 객체로 바꿀때 사용
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    labelMain.imgtk = imgtk
    labelMain.configure(image=imgtk)
    labelMain.after(10, detectAndDisplay)

    global writer
    if writer is None and output_name is not None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        fps = cap.get(cv.CAP_PROP_FPS)
        writer = cv.VideoWriter(output_name, fourcc, fps, (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)



writer = None

main = Tk()
main.title("YoLo detect")
main.geometry()


# 윈도우 font와 title_name 출력
sizeLable = Label(main, text='Frame Width/confidence: ').grid(row=1, column=0)
sizeValue = IntVar(value=frame_width)
sizeSpin = Spinbox(main, textvariable=sizeValue, from_=0, to=2000, increment=100, justify=RIGHT)
sizeSpin.grid(row=1, column=1)
confidence = DoubleVar(value=min_confidence)
sizeSpin2 = Spinbox(main, textvariable=confidence, from_=0, to=1, increment=0.05, justify=RIGHT)
sizeSpin2.grid(row=1, column=2)

Button(main, text='File Select', height=2, command=lambda: selectFile()).grid(row=1, column=3)

imageFrame = Frame(main)
imageFrame.grid(row=2, column=0, columnspan=4)
labelMain = Label(imageFrame)
labelMain.grid(row=0, column=0)

main.mainloop()
