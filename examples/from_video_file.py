import face_recognition
import cv2
import numpy as np

# Este demo tiene algunas modificaciones 
# para hacerlo mas rapido: 
# 1. Procesa video a 1/4 resolucion (más rapidez) y lo muestra a full
# 2. Solo detecta caras in cada frame
# 
# Nota: Requiere OpenCV (cv2 lib) para leer video
# aunque OpenCV no se requiere para usar la libreria face_recognition


# Esto se puede cambiar por video de una webcam o video streaming de un rtsp
# input_movie = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
input_movie = cv2.VideoCapture("http://192.168.0.9:8080/video/mjpeg")

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))


# Output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('outputs/output.avi', fourcc, 29.97, (640, 360))

# Cargar imagen(es) para reconocer en el video
matthew_face = face_recognition.load_image_file("pics/matthew.jpg")
eli_face = face_recognition.load_image_file("pics/eli.jpg")

# Este es el encoding que sacamos de la cara, algo asi como el 
# modelo biometrico de Neurotech que guardamos en un BLOB en la bd
matthew_encoding = face_recognition.face_encodings(matthew_face)[0]
eli_encoding = face_recognition.face_encodings(eli_face)[0]

# Array de uno o mas encodings
known_face_encodings = [
    matthew_encoding,
    eli_encoding
]

known_face_names = [
    "Matthew",
    "Eli"
]

# Some vars
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

# Procesar cada X frames, nos ayuda a aumentar velocidad
# y con un framerate de 150.0, no importa mucho
process_frame_every = 5
frames_not_processed = 0

while True:
    # Un solo frame del video
    ret, frame = input_movie.read()
    frame_number += 1
    
    # Detenerse si se acabo el video
    if not ret:
        print("Se acabo video")
        print("Frame stuck {}".format(frame))
        break

    # Cambiamos tamaño de frame para procesar mas rapido
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convertir de color BRG (opencv) a RGB (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Procesar cada X frames, nos ayuda a aumentar velocidad
    # y con un framerate de 150.0, no importa mucho
    if frames_not_processed == process_frame_every:
        face_locations = face_recognition.face_locations(rgb_small_frame)

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            #     name = known_face_names[best_match_index]

            face_names.append(name)
    

    # Logica para procesar solo cada X frames
    if frames_not_processed == process_frame_every: 
        frames_not_processed = 0
    else:
        frames_not_processed += 1
    
    # Resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        print("Drawing box in frame {}".format(frame_number))
        # Regresar a su tamaño normal
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        print("Rectangle in face drawn")


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write the resulting image to the output video file
    # print("Writing frame {} / {}".format(frame_number, length))
    # output_movie.write(frame)


print("Terminando video")
input_movie.release()
cv2.destroyAllWindows()