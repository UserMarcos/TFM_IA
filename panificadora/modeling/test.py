#from pathlib import Path

#from loguru import logger
#from tqdm import tqdm
import typer

from panificadora.config import MODELS_DIR, PROCESSED_DATA_DIR
from panificadora.modeling.utils import *


def test( nombre_modelo: str,
              batch_size: int = 32
            ):
                
    print("batch_size: ", batch_size)
    print("Carpeta con datos procesados: ", PROCESSED_DATA_DIR)
    print("Carpeta con modelos: ", MODELS_DIR)            
        
    model, nombre_fichero_pesos = get_modelo(nombre_modelo)     
    print(f"Creado el modelo {nombre_modelo}")
    
    datamodule = get_datamodule(batch_size)  
    print(f"Creado el 'datamodule' con {batch_size} como tamaño de batch")
    
    engine = get_engine()
    print("Clase engine creada")
    
    #engine.test(datamodule=datamodule, 
    #            model= model, 
    #            ckpt_path = MODELS_DIR / nombre_fichero_pesos
    #            )
       


if __name__ == "__main__":
    typer.run(test)
