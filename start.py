import face_recognition
import cv2
import time

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("./videos/JUAREZ Y PADRE MIER 1603.avi")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('./outputs/output.avi', fourcc, 150.00, (2592, 1944))

# Load some sample pictures and learn how to recognize them.
lmm_image = face_recognition.load_image_file("pics/demo1.png")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

# al_image = face_recognition.load_image_file("alex-lacamoire.png")
# al_face_encoding = face_recognition.face_encodings(al_image)[0]

known_faces = [
    lmm_face_encoding,
    # al_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

frameBatch = []

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1
    
    frameBatch.append(frame)

    # Quit when the input video file ends
    if not ret:
        break

    if len(frameBatch) == 128:
        print('in batch')
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        print("finding faces locations")
        # Find all the faces and face encodings in the current frame of video
        start_time = time.time()
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        elapsed_time = time.time() - start_time
        print("found, time elapsed: {}".format(elapsed_time))

        print("finding faces encodings")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        print('finding face names')
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

            # If you had more than 2 faces, you could make this logic a lot prettier
            # but I kept it simple for the demo
            name = None
            if match[0]:
                name = "Demo numero uno"
            # elif match[1]:
            #     name = "Alex Lacamoire"

            face_names.append(name)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            print("Working on frame {} / {}".format(frame_number, length))
            # if not name:
            #     continue
            name="Persona"
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)





            # Write the resulting image to the output video file
            print("Writing frame {} / {}".format(frame_number, length))
            # output_movie.write(frame)
            cv2.imwrite("./outputs/frame%d.jpg" % frame_number, frame)
            
        frameBatch = []
        print('cleared batch')

# All done!
# input_movie.release()
cv2.destroyAllWindows()