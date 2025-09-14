import os
from pathlib import Path

import cv2
from scipy.signal import find_peaks


def picos_valles(histograma):
    hist = histograma.flatten()
    distancia = 20
    prominencia = 20000
    
    # Buscar picos (máximos locales)
    picos, _ = find_peaks(hist, distance=distancia, prominence=prominencia)
    
    # Buscar valles (mínimos locales) invirtiendo la señal
    valles, _ = find_peaks(-hist, distance=distancia, prominence=prominencia)

    return picos, valles
    
def get_imagen_barra(imagen_tabla, contorno):
    x, y, w, h = cv2.boundingRect(contorno)

    if (x < 50): return None
    if (y < 50): return None
    if (x+w+50 > imagen_tabla.shape[1]) : return None
    if (y+h+50 > imagen_tabla.shape[0]) : return None
    if (w < 100): return None
    if (h < 100): return None
    if (h * w < 100000): return None
    # imagen_barra = imagen_1_canal[x-50:x+w+50, y-50:y+h+50, :]
    return imagen_tabla[y-50:y+h+50, x-50:x+w+50]


def grabar_imagenes_barra(ruta_fuente, ruta_destino):
    valor_maximo_umbral = 255
    lista_ficheros = os.scandir(ruta_fuente)
    
    for i, fichero in enumerate(lista_ficheros):
        if not fichero.is_file(): continue
        
        # Leer imagen con OpenCV (por defecto en BGR)
        imagen = cv2.imread(fichero)
        
        # Comprobar si se cargó correctamente
        if imagen is None: continue
        
        print(f"{i}: {fichero.name}")
        
        # Trabajamos solo con un canal
        imagen_1_canal = imagen[:, :, 0]
        
        histogram = cv2.calcHist([imagen_1_canal], [0], None, [255], [1, 255])
        peaks, valleys = picos_valles(histogram)
        
        umbral = valleys[-1] # valor de umbral
        
        _, imagen_umbralizada = cv2.threshold(imagen_1_canal, umbral, valor_maximo_umbral, cv2.THRESH_BINARY)
        
        #cv2.imwrite(ruta_destino/fichero.name, imagen_umbralizada)
        
        # Find contours
        contours, _ = cv2.findContours(imagen_umbralizada, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
                                        
        # Grabamos las barras en disco
        numero_barra = 0

        for contour in contours:
            imagen_barra = get_imagen_barra(imagen_1_canal, contour)
            if imagen_barra is None: continue
            cv2.imwrite(ruta_destino/f"barra{i:03<}_{numero_barra:03}.jpg", imagen_barra)
            numero_barra = numero_barra + 1         
          
        print(f"\tUmbral: {umbral}\tNúmero de barras: {numero_barra}")
           
        #if i>3: break
        
if __name__ == "__main__":
    origen = Path("data/raw/mis_imagenes/Dataset/Frimar/bijou")
    destino = Path("data/interim/Frimar/bijou/train")
    
    os.makedirs(destino, exist_ok=True)
    
    grabar_imagenes_barra(origen, destino)
