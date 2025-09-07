"""Getting Started with Anomalib Inference using the Python API.

This example shows how to perform inference on a trained model
using the Anomalib Python API.
"""

from anomalib.engine import Engine
#from torchvision.transforms import v2
#from pathlib import Path
#from anomalib.data import Folder
from datamodule_folder import get_datamodule, get_modelo, get_engine
from anomalib.metrics import AUROC, AUPRO, AUPR


if __name__ == '__main__':

    model = get_modelo()

    print("Modelo creado")
        
    engine = get_engine()

    print("Engine creado")
      
    datamodule = get_datamodule()
    
    print("Datamodule creado")

    #datamodule.setup(stage = "fit")
    #mi_DataLoader = datamodule.train_dataloader()

    #datamodule.setup(stage = "test")
    #mi_DataLoader = datamodule.test_dataloader()

    #print("DataLoader creado")

    prediciones = engine.predict(model=model, 
                                datamodule = datamodule,
                                #dataloaders = mi_DataLoader,
                                ckpt_path = "results/Cflow/Bijou/latest/weights/lightning/model.ckpt"
                                )
    
    print("Prediciones creadas tipo:", type(prediciones))
    print("Cantidad de batch:", len(prediciones))
    batch = prediciones[0]
    print("Formato del batch: ", batch.image.shape)
    print("Tamaño del batch: ", batch.batch_size)
    print("Tamaño de 'image_path':", len(batch.image_path))
    print("Valor de 'image_path':", batch.image_path[0])
    print("Tipo de 'anomaly_map':", type(batch.anomaly_map))
    print("Formato de 'anomaly_map':", batch.anomaly_map.shape)
    print("Tipo de 'pred_label':", type(batch.pred_label))
    print("Formato de 'pred_label':", batch.pred_label.shape)
    print(f"Valores de 'pred_label': {batch.pred_label[0]}, {batch.pred_label[1]}, {batch.pred_label[2]}...")
    print("Tipo de 'pred_score':", type(batch.pred_score))
    print("Formato de 'pred_score':", batch.pred_score.shape)
    print(f"Valores de 'pred_score': {batch.pred_score[0]}, {batch.pred_score[1]}, {batch.pred_score[2]}...")
    

    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    image_aupr = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    pixel_aupr = AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_")


    # name that will be used by Lightning when logging the metrics
    print(image_auroc.name)  # 'image_AUROC'
    print(image_aupr.name)
    print(pixel_auroc.name)  # 'pixel_AUROC'
    print(pixel_aupr.name)

    for batch in prediciones:
        image_auroc.update(batch)
        image_aupr.update(batch)
        pixel_auroc.update(batch)
        pixel_aupr.update(batch)

    print("Puntuación AUROC imagen:", image_auroc.compute())
    print("Puntuación AUPR imagen:", image_aupr.compute())
    print("Puntuación AUROC pixel:", pixel_auroc.compute())
    print("Puntuación AUPR pixel:", pixel_aupr.compute())

    '''
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
    '''