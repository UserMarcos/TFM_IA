import anomalib
from anomalib.data import Folder
from pathlib import Path
import os
import torch
from torchvision.transforms import v2
#from anomalib.models import Patchcore
from anomalib.models import Cflow
from anomalib.engine import Engine
#from anomalib.data import MVTecAD

# python_env\anomalib\Scripts\activate
# cd Documents\GitHub\TFM_IA
# make probar_anomalib

if __name__ == '__main__':
    print("Versión de anomalib: ",anomalib.__version__)
    print("Es compatible con cuda:", torch.cuda.is_available())
    print("Es compatible con xpu:", torch.xpu.is_available())

    carpeta = Path("data/processed/bijou2")
    #carpeta = Path("data/processed/MVTec-AD")
    print("Carpeta: ", carpeta)
    lista_ficheros = os.scandir(carpeta)

    for i, fichero in enumerate(lista_ficheros):
        print(f"{i:>2d}: {fichero.name}")
        
    # Creamos el data agumentation
    augmentations_train = v2.Compose([
        v2.Pad(50, 255, "edge"),
        v2.CenterCrop(256), 
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip()   
    ])
    
    augmentations_valid = v2.Compose([
        v2.Pad(50, 255, "edge"),
        v2.CenterCrop(256)   
    ])
    
    datamodule = Folder(
        name="Bijou",
        root=carpeta,
        normal_dir="train2",
        abnormal_dir="Barra_brillo",
        mask_dir = "mascara",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers = 1,
        train_augmentations = augmentations_train,
        val_augmentations = augmentations_valid,
        test_augmentations = augmentations_valid,
        augmentations = None
    )
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
    
    model = Cflow(
        #backbone="resnet18",
        #layers=["layer1", "layer2", "layer3"],
        #coreset_sampling_ratio=0.1,
        #pre_processor = exportable_transform
    )
    
    print(model.trainer_arguments)
    
    engine = Engine(
        max_epochs=1,  # Override default trainer settings
        #input_size = [128, 128]
        #precision="16-mixed"
        #default_root_dir=Path("./mis_resultados")
    )
    
    print("Clase engine creada")
    
    engine.fit(datamodule=datamodule, model=model)

