#from pathlib import Path

#from loguru import logger
#from tqdm import tqdm
import typer

from panificadora.config import MODELS_DIR, PROCESSED_DATA_DIR
from panificadora.modeling.utils import *


def entrenar( nombre_modelo: str,
              batch_size: int = 32,
              epocas: int = 10
            ):
                
    print("batch_size: ", batch_size)
    print("Nº de épocas: ", epocas)
    print("Carpeta con datos procesados: ", PROCESSED_DATA_DIR)
    print("Carpeta con modelos: ", MODELS_DIR)            
    
    model, nombre_fichero_pesos = get_modelo(nombre_modelo)
    
    if model is None:
        print("No hay modelo. Terminamos")
        quit()
        
    print(f"Creado el modelo {nombre_modelo}")
    
    datamodule = get_datamodule(batch_size)
    
    print(f"Creado el 'datamodule' con {batch_size} como tamaño de batch")
    
    engine = get_engine(epocas, nombre_fichero_pesos)
    print(f"Clase engine creada con {epocas} épocas y '{nombre_fichero_pesos}' como fichero de pesos")
    
    #engine.fit(datamodule=datamodule, model=model)


if __name__ == "__main__":
    typer.run(entrenar)
