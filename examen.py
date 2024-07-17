import cv2
import os
import face_recognition as fr
import numpy as np
from figura import Figura

radio = float(input("Introduce el radio: "))


ruta = "Empleados"
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)

for empleado in lista_empleados:
    imagen_actual = cv2.imread(f"{ruta}/{empleado}")
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(empleado)[0])


def codificar(imagenes):
    lista_codificada = []

    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        codificado = fr.face_encodings(imagen)
        if codificado:
            lista_codificada.append(codificado[0])
    return lista_codificada

lista_empleados_codificada = codificar(mis_imagenes)


captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)


exito, imagen = captura.read()


if not exito:
    print("No se pudo tomar la foto")
else:
    cara_captura = fr.face_locations(imagen)
    if not cara_captura:
        print("No se encontraron caras en la imagen capturada")
    else:
        cara_captura_codificada = fr.face_encodings(imagen, known_face_locations=cara_captura)
        for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):
            coincidencias = fr.compare_faces(lista_empleados_codificada, caracodif, tolerance=0.6)
            distancias = fr.face_distance(lista_empleados_codificada, caracodif)

            indice_coincidencia = np.argmin(distancias)

            if coincidencias[indice_coincidencia]:
                print(f"Bienvenido {nombres_empleados[indice_coincidencia]}")
                figura = Figura(radio)
                print(f"Área del círculo: {figura.area_circulo():.2f}")
                print(f"Área de la esfera: {figura.area_esfera():.2f}")
            else:
                print("No hay coincidencias")
                figura = Figura(radio)
                print(f"Volumen de la esfera: {figura.volumen_esfera():.2f}")

captura.release()
cv2.destroyAllWindows()
