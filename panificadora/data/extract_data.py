import py7zr
from tqdm import tqdm
from pathlib import Path
import os

class ExtractProgressBar(py7zr.callbacks.ExtractCallback, tqdm):
    def __init__(self, *args, total_bytes, **kwargs):
        super().__init__(self, *args, total=total_bytes, **kwargs)
    def report_start_preparation(self):  pass
    def report_start(self, processing_file_path, processing_bytes): pass
    def report_end(self, processing_file_path, wrote_bytes):
        #print()
        self.update(int(wrote_bytes))
    def report_postprocess(self): pass
    def report_warning(self, message):  pass
    def report_update(self, info): pass
    
def descomprimir(nombre_fichero, carpeta_salida):
    with py7zr.SevenZipFile(nombre_fichero, 'r') as archive:
        archive_info = archive.archiveinfo()
        with ExtractProgressBar(
                 unit='B',
                 unit_scale=True,
                 miniters=1,
                 total_bytes=archive_info.uncompressed,
                 desc=f"Descomprimiendo") as progress:
            archive.extractall(path=carpeta_salida, callback=progress)

if __name__ == "__main__":
    destino = Path("data/raw/mis_imagenes")
    
    # Unimos la ruta y el nombre del fichero
    fichero_7z = destino / "DatasetB.7z"
    if (fichero_7z.exists()): 
        descomprimir(fichero_7z, destino)
        print(f"Archivo Dataset.7z descomprimido en {os.path.abspath(destino)}")
    else:
        print(f"El fichero no existe")
    
