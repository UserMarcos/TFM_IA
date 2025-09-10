from anomalib.data import Folder
from torchvision.transforms import v2
from pathlib import Path
from anomalib.metrics import Evaluator, AUROC, AUPR, AUPRO
from anomalib.models import Cflow, EfficientAd, Dsr, Patchcore
#from anomalib.loggers import AnomalibMLFlowLogger
from lightning.pytorch.loggers import TensorBoardLogger
from anomalib.engine import Engine
from anomalib.callbacks import TilerConfigurationCallback
import warnings

def get_datamodule(batch_size):
    carpeta = Path("data/processed/bijou")
    #carpeta = Path("data/processed/MVTec-AD")
    print("Carpeta del dataset: ", carpeta)
    '''
    lista_ficheros = os.scandir(carpeta)

    for i, fichero in enumerate(lista_ficheros):
        print(f"{i:>2d}: {fichero.name}")
    '''
        
    # Creamos el data agumentation
    augmentations_train = v2.Compose([
        #v2.Grayscale(num_output_channels=1),
        v2.Pad(50, 255, "edge"),
        v2.CenterCrop(256), 
        #v2.Resize(128),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip()   
    ])
    
    augmentations_valid = v2.Compose([
        #v2.Grayscale(num_output_channels=1),
        v2.Pad(50, 255, "edge"),
        v2.CenterCrop(256),
        #v2.Resize(128)
    ])
    
    datamodule_folder = Folder(
        name="Bijou_b",
        root=carpeta,
        normal_dir="train",
        abnormal_dir="test",
        mask_dir = "mascara",
        train_batch_size= batch_size, #8, # en PatchCore = 8
        eval_batch_size = batch_size,  #
        #num_workers = 1,
        train_augmentations = augmentations_train,
        val_augmentations = augmentations_valid,
        test_augmentations = augmentations_valid,
        augmentations = None
    )
    
    return datamodule_folder

def get_evaluator():
   # Validation metrics
    val_metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),     # Image-level AUROC
        AUPR(fields=["pred_score", "gt_label"], prefix="image_")
        #F1Score(fields=["pred_label", "gt_label"])    # Image-level F1
    ]

    # Test metrics (more comprehensive)
    test_metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),     # Image-level AUROC
        AUPR(fields=["pred_score", "gt_label"], prefix="image_"),
        #AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),     # Pixel-level AUROC
        #AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),     # Pixel-level AUPRO
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        #F1Score(fields=["pred_label", "gt_label"]),   # Image-level F1
        #F1Score(fields=["pred_mask", "gt_mask"])      # Pixel-level F1
    ]


    # Create evaluator with both sets
    evaluador = Evaluator(
        val_metrics=val_metrics,
        test_metrics=test_metrics
    )

    return evaluador

def get_modelo_CFlow():

    # Create evaluator with both sets
    evaluator = get_evaluator()

    modelo = Cflow(
        #backbone="resnet18",
        #layers=["layer1", "layer2", "layer3"],
        #coreset_sampling_ratio=0.1,
        #pre_processor = exportable_transform
        evaluator = evaluator
    )
    return modelo

def get_modelo_Dsr():
    warnings.filterwarnings(
        "ignore",
        message="Attribute 'evaluator' is an instance of `nn.Module`.*",
    )
    # Create evaluator with both sets
    evaluator = get_evaluator()

    modelo = Dsr(
        #evaluator = evaluator
    )
    
    modelo.save_hyperparameters(ignore=["evaluator"])
    
    print(type(modelo.model))
    
    return modelo

def get_modelo_EfficientAd():

    # Create evaluator with both sets
    evaluator = get_evaluator()

    modelo = EfficientAd(
        #backbone="resnet18",
        #layers=["layer1", "layer2", "layer3"],
        #coreset_sampling_ratio=0.1,
        #pre_processor = exportable_transform
        evaluator = evaluator
    )
    return modelo

def get_modelo_PatchCore():

    # Create evaluator with both sets
    evaluator = get_evaluator()

    modelo = Patchcore(
        backbone= "resnet18",
        layers=["layer3"],
        #coreset_sampling_ratio=0.1,
        #pre_processor = exportable_transform
        num_neighbors = 3,
        evaluator = evaluator
    )
    return modelo

def get_engine(epocas):

    #mlflow_logger = AnomalibMLFlowLogger()
    logger = TensorBoardLogger("tb_logs", name="anomalib_experiment")
    #tiler_config_callback = TilerConfigurationCallback(enable=True, tile_size=64, stride=64)

    motor = Engine(
        max_epochs=epocas,  # Override default trainer settings
        logger=logger,
        #check_val_every_n_epoch=5
        #input_size = [128, 128]
        precision= "bf16-mixed", #"16-mixed",
        #callbacks=[tiler_config_callback]
        #default_root_dir=Path("./mis_resultados")
        #accelerator="cpu"
    )

    return motor


if __name__ == '__main__':
    datamodule = get_datamodule()
