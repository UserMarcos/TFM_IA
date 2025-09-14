import base64
import os
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import py7zr
import requests
from tqdm import tqdm
import typer

from panificadora.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; OneDriveDownloader/1.0)"
}

def append_download_param(url: str) -> str:
    """Añade download=1 respetando si ya hay query en la URL."""
    parsed = urlparse(url)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    q["download"] = "1"
    new_query = urlencode(q)
    return urlunparse(parsed._replace(query=new_query))

def sharepoint_download_share(url: str) -> str:
    """
    Construye la URL de descarga directa usando:
    https://<dominio>/_layouts/15/download.aspx?share=<base64urlsafe(sin =)>
    """
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    token = base64.b64encode(url.encode("utf-8")).decode("ascii")
    token = token.rstrip("=").replace("+", "-").replace("/", "_")
    return f"{origin}/_layouts/15/download.aspx?share={token}"

def filename_from_cd(content_disposition: str) -> str | None:
    if not content_disposition:
        return None
    # filename*=UTF-8''...  (RFC 5987)
    m = re.search(r"filename\*\s*=\s*UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
    if m:
        from urllib.parse import unquote
        return os.path.basename(unquote(m.group(1)))
    # filename="..."
    m = re.search(r'filename\s*=\s*"([^"]+)"', content_disposition, re.IGNORECASE)
    if m:
        return os.path.basename(m.group(1))
    # filename=...
    m = re.search(r'filename\s*=\s*([^;]+)', content_disposition, re.IGNORECASE)
    if m:
        return os.path.basename(m.group(1).strip())
    return None

def is_html_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "")
    return "text/html" in ctype.lower()

def try_download(url: str, out_path: str | None = None) -> str:
    """
    Descarga desde `url` probando:
      1) url con download=1
      2) /_layouts/15/download.aspx?share=<token>
    Devuelve la ruta final del archivo guardado.
    """
    with requests.Session() as s:
        s.headers.update(SESSION_HEADERS)

        # Intento 1: forzar download=1
        direct1 = append_download_param(url)
        resp = s.get(direct1, allow_redirects=True, stream=True, timeout=60)
        if resp.ok and not is_html_response(resp) and resp.headers.get("Content-Length", "0") != "0":
            return save_stream(resp, out_path)

        # Intento 2: endpoint download.aspx?share=
        direct2 = sharepoint_download_share(url)
        resp = s.get(direct2, allow_redirects=True, stream=True, timeout=60)
        if resp.ok and not is_html_response(resp) and resp.headers.get("Content-Length", "0") != "0":
            return save_stream(resp, out_path)

        # Si llega aquí, probablemente requiere autenticación o el link no es público
        raise RuntimeError(
            f"No se pudo descargar. Código HTTP={resp.status_code}. "
            "Es posible que el enlace requiera iniciar sesión o no sea 'cualquiera con el enlace'."
        )

def save_stream(resp: requests.Response, out_path: str | None) -> str:
    # Determinar nombre de archivo
    cd = resp.headers.get("Content-Disposition", "")
    fname = filename_from_cd(cd)
    if not fname:
        # fallback: usar la última parte de la URL o nombre genérico
        tail = urlparse(resp.url).path.split("/")[-1]
        fname = tail if tail else "archivo_onedrive"
    if out_path:
        if os.path.isdir(out_path):
            file_path = os.path.join(out_path, fname)
        else:
            file_path = out_path
    else:
        file_path = fname
        
    # Tamaño total
    total = int(resp.headers.get("Content-Length", 0))   

    # Guardar
    with open(file_path, "wb") as f, tqdm(
        total= total if total > 0 else None, 
        unit="B", 
        unit_scale=True, 
        desc=f"Descargando {fname}"
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 15):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return file_path
    

@app.command()
def descargar(
    url_zip: str = "https://alumnosviu-my.sharepoint.com/:u:/g/personal/mcastroc_student_universidadviu_com/ERMGeDVZNbVJmQ9CEI-rebUBb3ZyQ1XCSO3ltB_pg1BFZA?e=iUa3Gx",
    destino: Path = RAW_DATA_DIR / "bijou.7z"
):
    if destino.exists() and not destino.is_dir():
        print(f"El fichero '{destino}' ya estaba descargado")
        return destino
    #destino.mkdir(parents=True, exist_ok=True)
    #print("Carpeta creada")
    #descargar_y_extraer_zip(url_zip, destino)
    ruta = try_download(url_zip, destino)
    print("Descargado en:", ruta)
    return ruta
    
    
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


@app.command()
def descomprimir(
    nombre_fichero: Path = RAW_DATA_DIR / "bijou2.7z", 
    carpeta_salida: Path = PROCESSED_DATA_DIR / "bijou_extraido"
):
    if (not nombre_fichero.exists()):
        print(f"El fichero '{nombre_fichero}' no existe. Prueba a descargar primero")
        return
        
    # Vemos si ya fue descomprimido
    if Path(carpeta_salida).exists():
        print(f"El fichero ya fue descomprimido en '{carpeta_salida}' previamente")
        return
        
    with py7zr.SevenZipFile(nombre_fichero, 'r') as archive:
        archive_info = archive.archiveinfo()
        with ExtractProgressBar(
                 unit='B',
                 unit_scale=True,
                 miniters=1,
                 total_bytes=archive_info.uncompressed,
                 desc="Descomprimiendo") as progress:
            archive.extractall(path=carpeta_salida, callback=progress)
    print(f"El fichero '{nombre_fichero}' descomprimido en '{carpeta_salida}'")

@app.command()   
def get_dataset():
    ruta = descargar()
    descomprimir(ruta)
    
    #with open(ruta, "r") as f: data = f.read()
    os.remove(ruta)
    

if __name__ == "__main__":
    app()
