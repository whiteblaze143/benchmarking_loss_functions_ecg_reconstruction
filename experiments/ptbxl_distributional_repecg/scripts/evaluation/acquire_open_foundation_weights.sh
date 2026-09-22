#!/usr/bin/env bash
# Acquire only public, checksum-published ECG foundation-model weights to NFS.
set -euo pipefail

root=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_weights

download() {
    local group=$1 name=$2 md5=$3 url=$4
    local directory="$root/$group"
    local target="$directory/$name"
    local partial="$target.part"
    mkdir -p "$directory"
    if test -e "$target"; then
        echo "refusing to overwrite existing artifact: $target" >&2
        exit 1
    fi
    if test -e "$partial"; then
        curl --fail --location --retry 5 --continue-at - --output "$partial" "$url"
    else
        curl --fail --location --retry 5 --output "$partial" "$url"
    fi
    printf '%s  %s\n' "$md5" "$partial" | md5sum --check --status -
    mv "$partial" "$target"
    printf 'verified %s\n' "$target"
}

download clef clef_small.ckpt de76ebbf1024e5774309c1d12d6970eb \
    https://zenodo.org/api/records/17572734/files/clef_small.ckpt/content
download clef clef_medium.ckpt 781eed16d190ef97733994cd2d01d1ff \
    https://zenodo.org/api/records/17572734/files/clef_medium.ckpt/content
download clef clef_largel.ckpt 07c494e85c9c17ddf34d8e679e02b0e0 \
    https://zenodo.org/api/records/17572734/files/clef_largel.ckpt/content
download ecgfm_ked best_valid_all_increase_with_augment_epoch_3.pt f5d5711b4fd52f41ac2e3c9eb10e4650 \
    https://zenodo.org/api/records/14881564/files/best_valid_all_increase_with_augment_epoch_3.pt/content

printf '[Finished] verified CLEF and ECGFM-KED public weights\n'
