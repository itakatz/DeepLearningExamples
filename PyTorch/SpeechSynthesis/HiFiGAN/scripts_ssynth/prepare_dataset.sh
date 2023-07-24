#!/usr/bin/env bash

set -e

#: ${DATASET_PATH:=data/LJSpeech-1.1}
: ${N_MEL_FILTERS:=80}
: ${DATASET_PATH:=data_ssynth}
: ${USE_SYNTH_WAVS:=false}
: ${SYNTH_SUFFIX:=synth}

if [ -f "$DATASET_PATH/metadata.csv" ]; then
    echo "File $DATASET_PATH/metadata.csv exists, not creating a new one"
else
    # create the metadata file
    find $DATASET_PATH/wavs/ -type f -name "*.wav" -print0 | xargs -0 basename -a -s '.wav' | sort | awk '{print $1"||"}' > $DATASET_PATH/metadata.csv
fi

export DATASET_PATH
export SYNTH_SUFFIX
bash scripts_ssynth/generate_filelists.sh

# Generate mel-spectrograms
# NOTE in training, the mel of target wav is not loaded from disk, only mel of source (which can be the same file but used once as input and once as target for loss)
#       so we don't need to create both mels - from original AND synthesized wavs. only one of them - the one that will be used as source for hifigan training
if [ "$USE_SYNTH_WAVS" != true ]; then
    python prepare_dataset.py \
        --wav-text-filelists $DATASET_PATH/filelists/ssynth_audio_$SYNTH_SUFFIX.txt \
        --n-workers 32 \
        --batch-size 1 \
        --dataset-path $DATASET_PATH \
        --extract-mels \
        --sampling-rate 44100 \
        --mel-fmax 22050 \
        --n-mel-channels $N_MEL_FILTERS \
        --max-wav-value 2147483648 \
        --use-mel-impl-from-hifigan
        "$@"
else
    #--- if we need mels from synthesized audio, run again with the flag --use-synthesized-wavs
    python prepare_dataset.py \
        --wav-text-filelists $DATASET_PATH/filelists/ssynth_audio_$SYNTH_SUFFIX.txt \
        --use-synthesized-wavs \
        --synth-suffix $SYNTH_SUFFIX \
        --n-workers 32 \
        --batch-size 1 \
        --dataset-path $DATASET_PATH \
        --extract-mels \
        --sampling-rate 44100 \
        --mel-fmax 22050 \
        --n-mel-channels $N_MEL_FILTERS \
        --max-wav-value 2147483648 \
        --use-mel-impl-from-hifigan
        "$@"
fi
