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

# Mostrar la imagen
#cv2.imshow("Foto de Control", foto_control)
#cv2.imshow("Foto de Prueba", foto_prueba)

# Esperar a que se presione una tecla
cv2.waitKey(0)
cv2.destroyAllWindows()
