# cargar librerias
# 2024-07-01
import face_recognition as fr
import cv2

# cargar imagenes
foto_control= fr.load_image_file("FotoA.jpg")
foto_prueba= fr.load_image_file("FotoB.jpg")

# convertir la imagen a RGB
foto_control= cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba= cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

# Verificar si las imagenes estan en el formato correcto
print(type(foto_control), foto_control.shape)
print(type(foto_prueba), foto_prueba.shape)

# Mostrar la imagen
#cv2.imshow("Foto de Control", foto_control)
#cv2.imshow("Foto de Prueba", foto_prueba)

# Esperar a que se presione una tecla
cv2.waitKey(0)
cv2.destroyAllWindows()

# Localizar cara control
cara_control_locs = fr.face_locations(foto_control)
#print(cara_control_locs)

if cara_control_locs:
      lugar_cara_A = cara_control_locs[0]
      cara_cadificada_A = fr.face_encodings(foto_control,known_face_locations=[lugar_cara_A])[0]
else:
    print("No se encontro ninguna cara en la imagen de control")

cara_control_locsB = fr.face_locations(foto_prueba)
#print(cara_control_locs)

if cara_control_locsB:
      lugar_cara_B = cara_control_locsB[0]
      cara_cadificada_B = fr.face_encodings(foto_prueba,known_face_locations=[lugar_cara_B])[0]
else:
    print("No se encontro ninguna cara en la imagen de prueba")

#MOSTRAR RECTANGLE
cv2.rectangle(foto_control,
              (lugar_cara_A[3], lugar_cara_A[0]),
              (lugar_cara_A[1], lugar_cara_A[2]),
              (0, 255, 0),
              2)
cv2.rectangle(foto_prueba,
              (lugar_cara_B[3], lugar_cara_B[0]),
              (lugar_cara_B[1], lugar_cara_B[2]),
              (0, 255, 0),
              2)
cv2.imshow("Foto de Control", foto_control)
cv2.imshow("Foto de Prueba", foto_prueba)
cv2.waitKey(0)

# realizar comparacion
comparacion = fr.compare_faces([cara_cadificada_A], cara_cadificada_B, tolerance=0.6)
print(comparacion)

#mostrar las imagenes
cv2.imshow("Foto de Control", foto_control)
cv2.imshow("Foto de Prueba", foto_prueba)
cv2.waitKey(0)

# medida de la distancia
distancia = fr.face_distance([cara_cadificada_A], cara_cadificada_B)
print(distancia)