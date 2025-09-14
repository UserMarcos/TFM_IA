Getting started
===============

This is where you describe how to get set up on a clean install, including the
commands necessary to get the raw data (using the `sync_data_from_s3` command,
for example), and then how to make the cleaned, final data sets.

## Bajar el _dataset_

Descarga un fichero comprimido a `./data/raw` y lo descomprime.

```bash
make data
```

## Entrenar los modelos

Desde la carpeta raíz del proyecto:

```bash
python -m  panificadora.modeling 
```
