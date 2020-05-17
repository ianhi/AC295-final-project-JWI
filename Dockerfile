FROM jupyter/tensorflow-notebook

RUN pip install sidecar albumentations albumentations segmentation-models
RUN jupyter labextension install @jupyter-widgets/jupyterlab-sidecar
RUN rm -r work
COPY . .
