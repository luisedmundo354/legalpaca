# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training@sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9

ARG SOURCE_DATE_EPOCH
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    DS_BUILD_OPS=0 \
    HF_HUB_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=17 \
    PYTHONUNBUFFERED=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_OFFLINE=1

COPY training_image/requirements.lock /opt/training_image/requirements.lock
COPY training_image/image_contract.json /opt/training_image/image_contract.json
COPY training_image/runtime_contract.py /opt/training_image/runtime_contract.py
COPY --chmod=0555 training_image/bootstrap.py /opt/training_bootstrap/bootstrap.py

RUN /opt/conda/bin/python -c \
        'import os,re; value=os.environ.get("SOURCE_DATE_EPOCH", ""); assert re.fullmatch(r"[1-9][0-9]*", value), value' \
    && /opt/conda/bin/python -m pip install \
        --no-build-isolation \
        --no-cache-dir \
        --no-compile \
        --no-deps \
        --require-hashes \
        -r /opt/training_image/requirements.lock \
    && /opt/conda/bin/python /opt/training_image/runtime_contract.py \
        --contract /opt/training_image/image_contract.json \
        --output /opt/training_image/runtime_inventory.json

LABEL org.opencontainers.image.base.name="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training" \
      org.opencontainers.image.base.digest="sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9" \
      org.opencontainers.image.title="arr-retrieval-controlled-training" \
      org.opencontainers.image.training-contract.sha256="db4b2b307a56686054c2c04fbcebf5c133077765074ceef61a613c183a4b04ef"

WORKDIR /
ENTRYPOINT ["bash","-m","start_with_right_hostname.sh"]
CMD ["/bin/bash"]
