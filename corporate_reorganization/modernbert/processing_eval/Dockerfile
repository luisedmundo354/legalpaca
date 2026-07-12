# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training@sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9 AS java

ADD --checksum=sha256:5b4dc8817df13f88f9bfc434e5d018adb535889ff2fe0ccf758bcebcc216f394 \
    https://corretto.aws/downloads/resources/21.0.11.10.1/amazon-corretto-21.0.11.10.1-linux-x64.tar.gz \
    /tmp/corretto.tar.gz
RUN mkdir /opt/amazon-corretto-21 \
    && tar -xzf /tmp/corretto.tar.gz \
        --strip-components=1 \
        -C /opt/amazon-corretto-21

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training@sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9 AS sparse-builder

ARG SOURCE_PARENT_EPOCH
ENV SOURCE_DATE_EPOCH=${SOURCE_PARENT_EPOCH}

COPY processing_eval/build_requirements.lock /opt/build/build_requirements.lock
COPY processing_eval/requirements.lock /opt/build/requirements.lock
RUN test -n "${SOURCE_PARENT_EPOCH}" \
    && /opt/conda/bin/python -m pip install \
        --no-cache-dir \
        --no-compile \
        --no-deps \
        --require-hashes \
        -r /opt/build/build_requirements.lock \
    && /opt/conda/bin/python -m pip install \
        --no-cache-dir \
        --no-compile \
        --no-deps \
        --no-build-isolation \
        --require-hashes \
        --only-binary=pyjnius \
        --no-binary=pyserini \
        -r /opt/build/requirements.lock

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training@sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9

ARG SOURCE_PARENT_COMMIT
ARG SOURCE_PARENT_EPOCH
ARG SOURCE_PARENT_RFC3339
ARG BUILD_IDENTITY_SHA256
ENV SOURCE_DATE_EPOCH=${SOURCE_PARENT_EPOCH}

COPY --from=java /opt/amazon-corretto-21 /opt/amazon-corretto-21

ENV JAVA_HOME=/opt/amazon-corretto-21 \
    PATH=/opt/amazon-corretto-21/bin:${PATH} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONHASHSEED=17 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=sparse-builder /opt/conda/lib/python3.11/site-packages/pyserini \
     /opt/conda/lib/python3.11/site-packages/pyserini
COPY --from=sparse-builder /opt/conda/lib/python3.11/site-packages/pyserini-1.5.0.dist-info \
     /opt/conda/lib/python3.11/site-packages/pyserini-1.5.0.dist-info
COPY --from=sparse-builder /opt/conda/lib/python3.11/site-packages/jnius \
     /opt/conda/lib/python3.11/site-packages/jnius
COPY --from=sparse-builder /opt/conda/lib/python3.11/site-packages/jnius_config.py \
     /opt/conda/lib/python3.11/site-packages/jnius_config.py
COPY --from=sparse-builder /opt/conda/lib/python3.11/site-packages/pyjnius-1.7.0.dist-info \
     /opt/conda/lib/python3.11/site-packages/pyjnius-1.7.0.dist-info
COPY --from=sparse-builder /opt/conda/lib/python3.11/site-packages/setup_sdist.py \
     /opt/conda/lib/python3.11/site-packages/setup_sdist.py

WORKDIR /opt/program/modernbert
RUN mkdir retriever processing_eval

COPY retriever/__init__.py \
     retriever/artifacts.py \
     retriever/baseline_artifacts.py \
     retriever/bm25.py \
     retriever/data.py \
     retriever/e5_pack_artifact.py \
     retriever/evaluation.py \
     retriever/evaluator.py \
     retriever/hashing.py \
     retriever/markup.py \
     retriever/models.py \
     retriever/provenance.py \
     retriever/query_packing.py \
     retriever/query_views.py \
     retriever/rankers.py \
     retriever/regimes.py \
     retriever/staged_data.py \
     retriever/
COPY processing_eval/build_base_modernbert.py \
     processing_eval/build_context.py \
     processing_eval/build_context_manifest.json \
     processing_eval/evaluate_sm.py \
     processing_eval/image_contract.json \
     processing_eval/image_smoke.py \
     processing_eval/

RUN test -n "${SOURCE_PARENT_COMMIT}" \
    && test -n "${SOURCE_PARENT_EPOCH}" \
    && test -n "${SOURCE_PARENT_RFC3339}" \
    && test -n "${BUILD_IDENTITY_SHA256}" \
    && /opt/conda/bin/python processing_eval/image_smoke.py \
        --contract processing_eval/image_contract.json \
        --build-manifest processing_eval/build_context_manifest.json \
        --expected-build-identity-sha256 "${BUILD_IDENTITY_SHA256}" \
        --expected-source-parent-commit "${SOURCE_PARENT_COMMIT}" \
        --expected-source-parent-epoch "${SOURCE_PARENT_EPOCH}" \
        --expected-source-parent-rfc3339 "${SOURCE_PARENT_RFC3339}"

LABEL org.opencontainers.image.base.digest="sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9" \
      io.arr-retrieval.build-identity-sha256="${BUILD_IDENTITY_SHA256}" \
      io.arr-retrieval.source-parent-commit="${SOURCE_PARENT_COMMIT}" \
      io.arr-retrieval.source-parent-epoch="${SOURCE_PARENT_EPOCH}" \
      io.arr-retrieval.source-parent-rfc3339="${SOURCE_PARENT_RFC3339}"

ENTRYPOINT ["/opt/conda/bin/python", "/opt/program/modernbert/processing_eval/evaluate_sm.py"]
