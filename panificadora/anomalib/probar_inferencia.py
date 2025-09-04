"""Getting Started with Anomalib Inference using the Python API.

This example shows how to perform inference on a trained model
using the Anomalib Python API.
"""

from anomalib.models import Cflow
from anomalib.engine import Engine
from torchvision.transforms import v2
from pathlib import Path
from anomalib.data import Folder

if __name__ == '__main__':

    model = Cflow(
        #backbone="resnet18",
        #layers=["layer1", "layer2", "layer3"],
        #coreset_sampling_ratio=0.1,
        #pre_processor = exportable_transform
    )

    print("Modelo creado")
        
    engine = Engine(
        max_epochs=1,  # Override default trainer settings
        #input_size = [128, 128]
        #precision="16-mixed"
        #default_root_dir=Path("./mis_resultados")
    )

    print("Engine creado")

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
    
    carpeta = Path("data/processed/bijou2")
    
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
    
    print("Datamodule creado")

    prediciones = engine.predict(model=model, 
                                datamodule = datamodule, 
                                ckpt_path = "results/Cflow/Bijou/v0/weights/lightning/model.ckpt")
    
    print("Prediciones creadas tipo:", type(prediciones))
    
    batch = prediciones[0]
    print("Formato del batch: ", batch.image.shape)
    print("Tamaño del batch: ", batch.batch_size)
    
    # 5. Access the results
    if prediciones is not None:
        for i, prediction in enumerate(prediciones):
            image_path = prediction.image_path
            anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
            pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
            pred_score = prediction.pred_score  # Image-level anomaly score
            print(f"{i}: {image_path}")
            if i >= 3 :
                print(type(prediction))
                break