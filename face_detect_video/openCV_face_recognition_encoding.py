import cv2 as cv
import face_recognition
import pickle

dataset_pathes = ["./JWS/", "./KDW/", "./YYS/"]

names = ["JWS", "KDW", "YYS"]

number_images = 20
image_type = ".jpg"

encoding_file = "The_matrix02.pickle"

model_method = "cnn"

knownEncodings = []
knownNames = []

for (i, dataset_path) in enumerate(dataset_pathes):
    name = names[i]

    for idx in range(number_images):
        file_name = dataset_path + str(name) + str(idx + 1) + image_type
        print(file_name)

        image = cv.imread(file_name)
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_image, model=model_method)
        encodings = face_recognition.face_encodings(rgb_image, boxes)

        for encoding in encodings:
            print(file_name, name, encoding)
            knownEncodings.append(encoding)
            knownNames.append(name)

data = {"encodings": knownEncodings, "names": knownNames}

f = open(encoding_file, "wb")

f.write(pickle.dumps(data))
f.close()
