print("""
  ____              _______ _   ______ _       ___ __     __
 |  _ \            /__   __(_)/__   __(_) _ __|_| |\ \\\\  / /
 | |_) |_   _         | |   _    | |   _ | '__| | | \ \\\\/ /
 |  _ <| | | |        | | 0| |   | |  | || |    | |  \  \\\\
 | |_) | |_| |        | | /| |   | |  | || |    | | / /\ \\\\
 |____/ \__, |        |_| /|_|   |_|  |_||_|    |_|/_/  \_\\\\
         __/ |                                               
        |___/                           
""")


import cv2
import os
import shutil

# Importar el módulo face_mesh y drawing_utils de la biblioteca mediapipe.
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Ruta de la carpeta donde se guardarán las capturas.
capture_folder = 'captures'

# Eliminar la carpeta de capturas si existe.
if os.path.exists(capture_folder):
    shutil.rmtree(capture_folder)

# Crear la carpeta de capturas si no existe.
if not os.path.exists(capture_folder):
    os.mkdir(capture_folder)

# Inicializar el detector de rostros utilizando un clasificador haarcascade.
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar la cámara.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Lista de índices de puntos de interés en la malla de la cara.
index_list = [227, 116, 345, 447, 18, 200, 8, 9]

# Contador de capturas.
count = 1

# Inicializar el detector de malla de cara.
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.9) as face_mesh:

    # Bucle principal del programa.
    while True:
        # Leer un frame de la cámara.
        ret, frame = cap.read()
        if not ret:
            break
        
        # Invertir el frame horizontalmente para obtener una imagen espejo.
        frame = cv2.flip(frame, 1)

        # Obtener las dimensiones del frame.
        height, width, _ = frame.shape

        # Convertir el frame a RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar la malla de la cara en el frame.
        result = face_mesh.process(frame_rgb)

        # Copiar el frame para guardar una captura si es necesario.
        image_aux = frame.copy()

        # Convertir el frame a escala de grises.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en el frame.
        faces = face_classifier.detectMultiScale(gray, 1.2, 10)

        # Si se detecta al menos una cara en el frame.
        if result.multi_face_landmarks is not None:
            # Iterar sobre cada uno de los puntos de interés en la malla de la cara.
            for face_landmarks in result.multi_face_landmarks:
                for index in index_list:
                    # Obtener las coordenadas del punto de interés en el frame.
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)

                    # Dibujar un círculo en el punto de interés.
                    cv2.circle(frame, (x, y), 2, (253, 0, 253), 1)

                    # Guardar las coordenadas de los puntos de interés que se necesitarán más adelante.
                    if index == 227:
                        x_227 = x
                    elif index == 116:
                        x_116 = x
                    elif index == 345:
                        x_345 = x
                    elif index == 447:
                        x_447 = x
                        # Calcular la diferencia entre los ángulos laterales de las mejillas.
                        angle_diff_side = x_116 - x_227 - (x_447 - x_345)
                    elif index == 18:
                        x_18 = x
                    elif index == 8:
                        x_8 = x
                        # Calcular la diferencia entre el ángulo superior de la frente e inferior de la boca.
                        angle_diff_top_bottom = x_18 - x_8

                        # Mostrar la diferencia entre los ángulos laterales y superiores e inferiores.
                        print(f'    x: {angle_diff_side}\n    y: {angle_diff_top_bottom}\n')

                        # Si las diferencias entre los ángulos cumplen con ciertas condiciones, guardar una captura del rostro centrado.
                        if -3 <= angle_diff_side <= 3 and -3 <= angle_diff_top_bottom <= 3:
                            cv2.imwrite(f'{capture_folder}/i_{count}.jpg', image_aux)
                            print('Saved image')
                            # Iterar sobre los rostros detectados en el frame.
                            for (x, y, w, h) in faces:
                                # Aisla el rostro en 500x500px.
                                face = image_aux[y:y+h,x:x+w]
                                face = cv2.resize(face,(500,500), interpolation=cv2.INTER_CUBIC)
                                cv2.imwrite(f'{capture_folder}/r_{count}.jpg', face)
                                count +=1

        # Mostrar el frame con los puntos de interés dibujados.
        cv2.imshow('Video', frame)

        # Si se presiona la tecla 'q', salir del programa.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
