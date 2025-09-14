import argparse
import os

#import numpy as np
from pathlib import Path

import cv2

#from scipy.signal import find_peaks

def quitar_brillos_imagenes(ruta_fuente, ruta_destino, funcion_thresh):
    umbral = 220 #50 # valor de umbral
    lista_ficheros = os.scandir(ruta_fuente)
    
    for i, fichero in enumerate(lista_ficheros):
        if not fichero.is_file(): continue
        
        # Leer imagen con OpenCV (por defecto en BGR)
        imagen_disco = cv2.imread(fichero, cv2.IMREAD_GRAYSCALE)
        
        # Comprobar si se cargó correctamente
        if imagen_disco is None: continue
        
        print(f"{i}: {fichero.name}")
        
        _, imagen_umbralizada = cv2.threshold(imagen_disco, 
                                              umbral, 
                                              255, 
                                              funcion_thresh) #cv2.THRESH_TOZERO_INV)
                                        
        cv2.imwrite(ruta_destino/fichero.name, imagen_umbralizada) 
        #print(f"\tUmbral: {umbral}\tNúmero de barras: {numero_barra}")
        #if i>3: break
           
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para quitar o guardar los brillos de las imágenes de entrada.")

    parser.add_argument("-entrada", "--e", help="Carpeta de entrada")
    parser.add_argument("-salida", "--s", help="Carpeta de salida")
    parser.add_argument("-funcion", help="'mascara': graba máscara del brillo. 'quitar': graba la imagen sin brillo")

    args = parser.parse_args()
    print(vars(args))

    if not args.e:
        print("No hay carpeta de entrada")
        quit()

    origen = Path(args.e)
    #Comprobamos si es una carpeta
    if not origen.exists() or not origen.is_dir():
        print("Error: La ruta 'origen' no existe")
        quit()
    elif not origen.is_dir():
        print(f"Error: La ruta '{origen}' no es una carpeta")
        quit()


    if not args.s:
        print("No hay carpeta de salida")
        quit()

    destino = Path(args.s)

    if not args.funcion:
        print("No se especificó función a realizar")
        quit()

    if args.funcion == "mascara":
        funcion = cv2.THRESH_BINARY
    elif args.funcion == "quitar":
        funcion = cv2.THRESH_TOZERO_INV
    else:
        print("Valor de funcion no válido")
        quit()

    os.makedirs(destino, exist_ok=True)

    quitar_brillos_imagenes(origen, destino, funcion)    
