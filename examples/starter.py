import face_recognition
import cv2
import numpy as np

input_movie = cv2.VideoCapture("http://192.168.0.11:8080/video/mjpeg")

# input_movie = cv2.VideoCapture("videos/2020_05_16_14_00_00_BUL12503_3_14;00,_14;02.avi")
# length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = 0

matthew_face = face_recognition.load_image_file("pics/matthew.jpg")
eli_face = face_recognition.load_image_file("pics/eli.jpg")

matthew_encoding = face_recognition.face_encodings(matthew_face)[0]
eli_encoding = face_recognition.face_encodings(eli_face)[0]

known_face_encodings = [
    matthew_encoding,
    eli_encoding
]

known_face_names = [
    "Matthew",
    "Eli"
]

process_frame_every = 5
frames_not_processed = 0

while True:
    ret, frame = input_movie.read()
    frame_number += 1
    print("Frame number: {}".format(frame_number))
    
    if not ret:
        print("Video finished in frame {} / {}".format(frame_number, length))
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # This "if" only helps process every X frames, used for speed
    if frames_not_processed == process_frame_every:    
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                # first_match_index = matches.index(True)
                # name = known_face_names[first_match_index]

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
            print("Face found: " + name)
    
    if frames_not_processed == process_frame_every: 
        frames_not_processed = 0
    else:
        frames_not_processed += 1

    
input_movie.release()
cv2.destroyAllWindows()