# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM 371087393859.dkr.ecr.us-east-1.amazonaws.com/arr-retrieval-eval@sha256:00feb4550b52712901933a546a561c18896304e7d72109f0a5ce49220dd12cf2

WORKDIR /opt/program/modernbert

# Prove the inherited runtime before replacing the four source files whose
# current validators are required by fold evaluation.
RUN --network=none /opt/conda/bin/python processing_eval/image_smoke.py \
        --contract processing_eval/image_contract.json \
        --build-manifest processing_eval/build_context_manifest.json

ARG SOURCE_PARENT_COMMIT
ARG SOURCE_PARENT_EPOCH
ARG SOURCE_PARENT_RFC3339
ARG BUILD_IDENTITY_SHA256
ENV SOURCE_DATE_EPOCH=${SOURCE_PARENT_EPOCH}

COPY retriever/artifacts.py \
     retriever/evaluator.py \
     retriever/provenance.py \
     retriever/staged_data.py \
     retriever/
COPY processing_fold_eval/__init__.py \
     processing_fold_eval/archive_bridge.py \
     processing_fold_eval/build_context.py \
     processing_fold_eval/build_context_manifest.json \
     processing_fold_eval/evaluate_sm.py \
     processing_fold_eval/image_contract.json \
     processing_fold_eval/image_smoke.py \
     processing_fold_eval/inventory_sm.py \
     processing_fold_eval/

RUN --network=none test -n "${SOURCE_PARENT_COMMIT}" \
    && test -n "${SOURCE_PARENT_EPOCH}" \
    && test -n "${SOURCE_PARENT_RFC3339}" \
    && test -n "${BUILD_IDENTITY_SHA256}" \
    && /opt/conda/bin/python processing_fold_eval/image_smoke.py \
        --contract processing_fold_eval/image_contract.json \
        --build-manifest processing_fold_eval/build_context_manifest.json \
        --expected-build-identity-sha256 "${BUILD_IDENTITY_SHA256}" \
        --expected-source-parent-commit "${SOURCE_PARENT_COMMIT}" \
        --expected-source-parent-epoch "${SOURCE_PARENT_EPOCH}" \
        --expected-source-parent-rfc3339 "${SOURCE_PARENT_RFC3339}"

LABEL org.opencontainers.image.base.digest="sha256:00feb4550b52712901933a546a561c18896304e7d72109f0a5ce49220dd12cf2" \
      io.arr-retrieval-fold-eval.base-config-digest="sha256:76c29a7f5ca0a1a36d0f8b53fe1e49f40ab199f8ff1bc594ddbb09107c7749e8" \
      io.arr-retrieval-fold-eval.build-identity-sha256="${BUILD_IDENTITY_SHA256}" \
      io.arr-retrieval-fold-eval.source-parent-commit="${SOURCE_PARENT_COMMIT}" \
      io.arr-retrieval-fold-eval.source-parent-epoch="${SOURCE_PARENT_EPOCH}" \
      io.arr-retrieval-fold-eval.source-parent-rfc3339="${SOURCE_PARENT_RFC3339}"

ENTRYPOINT ["/opt/conda/bin/python", "/opt/program/modernbert/processing_fold_eval/evaluate_sm.py"]
