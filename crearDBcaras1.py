# crear base de datos de caras

import cv2
import os
import face_recognition as fr
import numpy

#ruta
ruta = "Empleados" #carpeta donde se encuentran las fotos de los empleados
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)

for empleaado in lista_empleados:
    imagen_actual = cv2.imread(f"{ruta}/{empleaado}")
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(empleaado)[0])

# print(nombres_empleados)

# funcion codificar imagenes para obtener las codificaciones de las caras

def codificar (imagenes):
    lista_codificada = []

    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) # convertir imagen a rgb
        codificado = fr.face_encodings(imagen)[0] # donde esta la cara
        lista_codificada.append(codificado) # agregar a la lista
    return lista_codificada

lista_empleados_codificada = codificar(mis_imagenes)
# print(len(lista_empleados_codificada))

# tomar una foto de la camara
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# leer foto
exito,imagen = captura.read()

# cv2.imshow("Captura", imagen)
# cv2.waitKey(0)

#print("Exito", exito)
# determinar si se pudo tomar la foto
if not exito:
    print("No se pudo tomar la foto")
else:
    #reconocer cara en foto
    cara_captura = fr.face_locations(imagen)
    #codificar cara
    cara_captura_codificada = fr.face_encodings(imagen, known_face_locations=cara_captura)
    """"zip(cara_captura_codificada, cara_captura): La funcion zip toma dos (o mas) listas y las empareja elemento a elemento.
    Esto significa que en cada iteracion, caracodif toma un valor de cara_captura_codificada y caraubic tomar el valor correspondiente de cara_captura."""

    for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):
        coincidencias = fr.compare_faces(lista_empleados_codificada, caracodif, tolerance=0.6)
        distancias = fr.face_distance(lista_empleados_codificada, caracodif)

    # indice de coincidencia
    indice_coincidencia = numpy.argmin(distancias)
    print(f"Indice de coincidencia {indice_coincidencia}")

    # mostrar resultados
    try: #manejar las excepciones si no se encuentra coincidencia
        if distancias[indice_coincidencia]:
            print(f"Bienvenido {nombres_empleados[indice_coincidencia]}")
        else:
            print("No hay coincidencias")
    except:
        print("No se encontraron coincidencias")


