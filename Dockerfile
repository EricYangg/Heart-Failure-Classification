FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

RUN mamba update --quiet --file /tmp/conda-linux-64.lock \
  && mamba clean --all -y -f \
  && fix-permissions "${CONDA_DIR}" \
  && fix-permissions "/home/${NB_USER}"
