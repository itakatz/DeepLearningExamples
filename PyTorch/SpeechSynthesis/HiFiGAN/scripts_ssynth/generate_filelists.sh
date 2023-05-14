#!/usr/bin/env bash

set -e

#: ${DATASET_PATH:=data/LJSpeech-1.1}
: ${DATASET_PATH:=data_ssynth}
: ${SYNTH_SUFFIX:=""}
echo "dataset path: $DATASET_PATH"
echo "synth suffix: $SYNTH_SUFFIX"

#--- use this to set the suffix added to the source folders
#--- for example:
#---    synth_h10 is sources created with 10 sawtooth harmonics
#---    synth_k16 is sources created with up-to-16 harmonics of sawtooth
#: ${SYNTH_SUFFIX1:=synth_10h}
#: ${SYNTH_SUFFIX2:=synth_16k}

# Generate filelists
python common/split_ssynth.py --filelists-path "${DATASET_PATH}/filelists" --metadata-path "${DATASET_PATH}/metadata.csv" --subsets train val test all
if [ "$SYNTH_SUFFIX" != "" ]; then
    python common/split_ssynth.py --filelists-path "${DATASET_PATH}/filelists" --metadata-path "${DATASET_PATH}/metadata.csv" --add-wavs-synth --synth-suffix "$SYNTH_SUFFIX" --subsets all
fi

#--- for out sax synth, we don't have transcript, and don't use pitch for HiFiGan. So don't create these files
#python common/split_ssynth.py --filelists-path "${DATASET_PATH}/filelists" --metadata-path "${DATASET_PATH}/metadata.csv" --add-transcript --subsets all  # used to extract ground-truth mels or pitch
#python common/split_ssynth.py --filelists-path "${DATASET_PATH}/filelists" --metadata-path "${DATASET_PATH}/metadata.csv" --add-pitch --add-transcript --subsets all  # used during extracting fastpitch mels
