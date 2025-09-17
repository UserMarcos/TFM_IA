#from pathlib import Path

#from loguru import logger
#from tqdm import tqdm
import typer

from panificadora.config import MODELS_DIR, PROCESSED_DATA_DIR
from panificadora.modeling.utils import *


def test( nombre_modelo: str,
              batch_size: int = 32,
              nVersion: int = 0
            ):
                
    print("batch_size: ", batch_size)
    print("Carpeta con datos procesados: ", PROCESSED_DATA_DIR)
    print("Carpeta con modelos: ", MODELS_DIR)            
        
    model, nombre_fichero_pesos = get_modelo(nombre_modelo, nVersion)     
    print(f"Creado el modelo {nombre_modelo}")
    
    datamodule = get_datamodule(batch_size)  
    print(f"Creado el 'datamodule' con {batch_size} como tamaño de batch")
    
    engine = get_engine()
    print("Clase engine creada")
    
    ckpt_path = MODELS_DIR / (nombre_fichero_pesos+".ckpt")
    print(ckpt_path)
    
    engine.test(datamodule=datamodule, 
                model= model, 
                ckpt_path = ckpt_path
                )
       


if __name__ == "__main__":
    typer.run(test)
