import cv2 as cv
import face_recognition
import pickle
import time

image_file = "./video/인턴 (The Intern, 2015) 캐릭터 예고편.mp4"
encoding_file = "The_matrix01.pickle"
unknown_name = "Unknown"
model_method = "hog"

output_name = "./video/output_" + model_method + "3.mp4"


def detectAndDisplay(frame):
    start_time = time.time()
    rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb_image, model=model_method)
    encodings = face_recognition.face_encodings(rgb_image, boxes)
    print("boxes : ", len(boxes))
    print("encodings : ", len(encodings))
    print(boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = unknown_name
        # print("number of matches : ")
        if True in matches:
            matchedIdxes = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxes:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            print("name: ", name)
            name = max(counts, key=counts.get)
        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):

        text = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        # 모르는얼굴 모자이크
        if name == unknown_name:
            # color = (0, 0, 255)
            # line = 1
            # name = ""
            w = right - left
            h = bottom - top
            mosaic_rate = 10
            face_img = frame[top:bottom, left:right]
            # 자른 이미지를 지정한 배율로 확대/축소하기
            face_img = cv.resize(face_img, (w // mosaic_rate, h // mosaic_rate))
            # 확대/축소한 그림을 원래 크기로 돌리기
            face_img = cv.resize(face_img, (w, h), interpolation=cv.INTER_AREA)
            # 원래 이미지에 붙이기
            frame[top:bottom, left:right] = face_img
        else:

            cv.rectangle(frame, (left, top), (right, bottom), color, line)

            cv.putText(frame, name, (left, text), cv.COLOR_BGR2RGB, 0.75, color, line)

        # 모르는 얼굴 빨간 네모
        # if name == unknown_name:
        #     color = (0, 0, 255)
        #     line = 1
        #     name = ""
        #
        # cv.rectangle(frame, (left, top), (right, bottom), color, line)
        #
        # cv.putText(frame, name, (left, text), cv.COLOR_BGR2RGB, 0.75, color, line)

    end_time = time.time()
    process_time = end_time - start_time
    print("process time is ", process_time)
    cv.imshow("face_recognition", frame)

    global writer
    if writer is None and output_name is not None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(output_name, fourcc, 24, (frame.shape[1], frame.shape[0]),True)

    if writer is not None:
        writer.write(frame)


data = pickle.loads(open(encoding_file, "rb").read())
cap = cv.VideoCapture(image_file)
writer = None

if not cap.isOpened():
    print("Error - Not open video capture")
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print("No More Frame....")
        break
    detectAndDisplay(frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cv.waitKey(0)
cv.destroyAllWindows()
