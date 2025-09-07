import anomalib
#from anomalib.data import Folder
#from pathlib import Path
#import os
import torch
#from torchvision.transforms import v2
#from anomalib.models import Patchcore
#from anomalib.models import Cflow
#from anomalib.engine import Engine
#from anomalib.data import MVTecAD
from datamodule_folder import get_datamodule, get_modelo_PatchCore, get_engine
import timm
# python_env\anomalib\Scripts\activate
# cd Documents\GitHub\TFM_IA
# make probar_anomalib

if __name__ == '__main__':
    print("Versión de anomalib: ",anomalib.__version__)
    print("Es compatible con cuda:", torch.cuda.is_available())
    print("Es compatible con xpu:", torch.xpu.is_available())
    
    #for model_name in timm.list_models(pretrained=True): print(model_name)

    datamodule = get_datamodule()


    '''
    datamodule = MVTecAD(
        root=carpeta,  # Path to download/store the dataset
        category="bottle",  # MVTec category to use
        train_batch_size=4,  # Number of images per training batch
        eval_batch_size=4,  # Number of images per validation/test batch
        num_workers=8,  # Number of parallel processes for data loading
    )

    
    datamodule.setup(stage = "fit")


    mi_DataLoader = datamodule.train_dataloader()
    batch = next(iter(mi_DataLoader))

    print("Formato del batch: ", batch.image.shape)
    print("Tamaño del batch: ", batch.batch_size)

    item = batch.items[23]
    print(type(item.image))
    print("Formato de la imagen: ", batch.items[23].image.shape)

    print(datamodule.train_augmentations)
    
    model = Patchcore(
        #backbone="wide_resnet50_2",
        #layers=["layer2", "layer3"],
        #coreset_sampling_ratio=0.1,
        #pre_processor = exportable_transform
    )
    '''
    
    model = get_modelo_PatchCore()
    
    print(model.trainer_arguments)
    
    engine = get_engine()
    print("Clase engine creada")
    
    engine.fit(datamodule=datamodule, model=model)

