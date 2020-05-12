# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=jupyter/tensorflow-notebook
FROM $BASE_CONTAINER

RUN pip install sidecar
RUN jupyter labextension install @jupyter-widgets/jupyterlab-sidecar
COPY . .
RUN rm -r work
