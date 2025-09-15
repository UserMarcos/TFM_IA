import anomalib
import os
import torch
from datamodule_folder import get_datamodule, get_modelo_CFlow, get_modelo_Dsr, get_engine
import argparse

# python_env\anomalib\Scripts\activate
# cd Documents\GitHub\TFM_IA
# make probar_anomalib

if __name__ == '__main__':
    print("Versión de anomalib: ",anomalib.__version__)
    print("Es compatible con cuda:", torch.cuda.is_available())
    print("Es compatible con xpu:", torch.xpu.is_available())

    parser = argparse.ArgumentParser(description="Script que entrena un modelo pasado por argumento")
    parser.add_argument("-modelo", help="'Cflow': Método Cflow-ad. 'DSR': Método dsr. 'EfficientAd'. 'PatchCore'")
    parser.add_argument('-bs', '--batch-size', 
                       default=32, 
                        type=int, 
                        metavar='B',
                        help='train batch size (default: 32)')
    parser.add_argument('-ep', '--epocas', 
                        type=int, 
                        default=25, 
                        metavar='N',
                        help='number of meta epochs to train (default: 25)')

    args = parser.parse_args()
    
    for nombre, valor in vars(args).items():
        print(f"\t{nombre}: {valor}")
    
    if not args.modelo:
        print("No se especificó modelo")
        quit()
        
    if args.modelo == 'Cflow':
        model = get_modelo_CFlow()
        ckpt_path = "results/Cflow/Bijou_b/latest/weights/lightning/model.ckpt"
    elif args.modelo == 'Dsr':
        model = get_modelo_Dsr()
        ckpt_path = "results/Dsr/Bijou_512/latest/weights/lightning/model.ckpt"
    elif args.modelo == 'EfficientAd':
        model = get_modelo_EfficientAd()
        ckpt_path = "results/EfficientAd/Bijou_b/latest/weights/lightning/model.ckpt"
    elif args.modelo == 'Patchcore':
        model = get_modelo_PatchCore()
        ckpt_path = "results/Patchcore/Bijou_b/latest/weights/lightning/model.ckpt"
    else:
        print("modelo no especificado")
        quit()
        
    print(f"Creado el modelo {args.modelo}")
      
    datamodule = get_datamodule(args.batch_size)
    
    print(f"Creado el 'datamodule' con {args.batch_size} como tamaño de batch")
    #print(model.trainer_arguments)
    
    engine = get_engine(args.epocas)
    print(f"Clase engine creada con {args.epocas} épocas")

    engine.test(datamodule=datamodule, 
                model= model, 
                ckpt_path = ckpt_path
                )
