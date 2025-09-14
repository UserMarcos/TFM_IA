#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = panificadora
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
## 	pipenv install
	dir




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Bajamos las imágenes de ondrive
.PHONY: download_data

download_data:
	$(PYTHON_INTERPRETER) panificadora/data/download_data.py
	
extract_data:
	$(PYTHON_INTERPRETER) panificadora/data/extract_data.py

guardar_imagenes_barra:
	$(PYTHON_INTERPRETER) panificadora/data/grabar_imagenes_barra.py
	
quitar_brillos_imagenes:
	$(PYTHON_INTERPRETER) panificadora/data/quitar_brillos_imagenes.py \
		--e data/interim/Frimar/bijou/train \
		-salida data/interim/Frimar/bijou/train3 \
		-funcion quitar

mascara_brillos:
	$(PYTHON_INTERPRETER) panificadora/data/quitar_brillos_imagenes.py \
		--e data/processed/bijou/test/Barra_brillo \
		-salida data/processed/bijou/mascara/Barra_brillo \
		-funcion mascara

probar_anomalib:
	$(PYTHON_INTERPRETER) panificadora/anomalib/pruebas.py \
		-modelo Dsr \
		-bs 8 \
		-ep 3
	
inferencia_anomalib:
	$(PYTHON_INTERPRETER) panificadora/anomalib/probar_inferencia.py

test_anomalib:
	$(PYTHON_INTERPRETER) panificadora/anomalib/probar_test.py

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	pipenv --python $(PYTHON_VERSION)
	@echo ">>> New pipenv created. Activate with:\npipenv shell"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m panificadora.dataset get-dataset


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
