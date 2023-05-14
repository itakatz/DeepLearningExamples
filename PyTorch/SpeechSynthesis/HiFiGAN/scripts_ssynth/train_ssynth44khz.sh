#!/usr/bin/env bash

export OMP_NUM_THREADS=1

# Enables faster cuDNN kernels (available since the 21.12-py3 NGC container)
export CUDNN_V8_API_ENABLED=1  # Keep the flag for older containers
export TORCH_CUDNN_V8_API_ENABLED=1
: ${MAX_WAV_VALUE:=2147483648}
: ${SAMPLING_RATE:=44100}
: ${N_MEL_FILTERS:=80}
: ${NUM_GPUS:=8}
: ${BATCH_SIZE:=16}
: ${AMP:=false}
: ${EPOCHS:=10000}
#: ${OUTPUT_DIR:="results/hifigan_ssynth44khz"}
: ${OUTPUT_DIR:="results/2023_01_20_hifigan_ssynth44khz_synthesized_input"}
: ${LOG_FILE:=$OUTPUT_DIR/nvlog.json}
: ${DATASET_DIR:="data_ssynth"}
: ${TRAIN_FILELIST:="$DATASET_DIR/filelists/ssynth_audio_train.txt"}
: ${VAL_FILELIST:="$DATASET_DIR/filelists/ssynth_audio_val.txt"}

: ${FINE_TUNE_DIR:=""}


: ${USE_SYNTH_WAVS:=false}
: ${SYNTH_SUFFIX:=synth}

if [ "$USE_SYNTH_WAVS" = true ]; then
    if [ "$FINE_TUNE_DIR" != "" ]; then
        echo "ERROR: cannot use both fine-tuning and synthesized wavs, as both use the argument --input_mels_dir"
        exit
    fi
    SYNTH_MEL_DIR="$DATASET_DIR"/mels_"$SYNTH_SUFFIX"
    #--- if num of mel filters is non-default, I add a suffix to mels folder
    if [ $N_MEL_FILTERS != 80 ]; then
        SYNTH_MEL_DIR="$SYNTH_MEL_DIR"_n"$N_MEL_FILTERS"
    fi
fi

# Intervals are specified in # of epochs
: ${VAL_INTERVAL:=10}
: ${SAMPLES_INTERVAL:=100}
: ${CHECKPOINT_INTERVAL:=10}
: ${LEARNING_RATE:=0.0003}
: ${LEARNING_RATE_DECAY:=0.9998}
: ${GRAD_ACCUMULATION:=1}
: ${RESUME:=true}


mkdir -p "$OUTPUT_DIR"

# Adjust env variables to maintain the global batch size:
#     NUM_GPUS x BATCH_SIZE x GRAD_ACCUMULATION = 128
GBS=$(($NUM_GPUS * $BATCH_SIZE * $GRAD_ACCUMULATION))
[ $GBS -ne 128 ] && echo -e "\nWARNING: Global batch size changed from 128 to ${GBS}."
echo -e "\nAMP=$AMP, ${NUM_GPUS}x${BATCH_SIZE}x${GRAD_ACCUMULATION}" \
        "(global batch size ${GBS})\n"

ARGS+=" --cuda"
ARGS+=" --dataset_path $DATASET_DIR"
ARGS+=" --training_files $TRAIN_FILELIST"
ARGS+=" --validation_files $VAL_FILELIST"
ARGS+=" --output $OUTPUT_DIR"
ARGS+=" --checkpoint_interval $CHECKPOINT_INTERVAL"
ARGS+=" --epochs $EPOCHS"
ARGS+=" --batch_size $BATCH_SIZE"
ARGS+=" --learning_rate $LEARNING_RATE"
ARGS+=" --lr_decay $LEARNING_RATE_DECAY"
ARGS+=" --validation_interval $VAL_INTERVAL"
ARGS+=" --samples_interval $SAMPLES_INTERVAL"
ARGS+=" --sampling_rate $SAMPLING_RATE"
ARGS+=" --mel_fmax 22050"
ARGS+=" --num_mels $N_MEL_FILTERS"

#--- in the dataset class hifigan/data_function.py:MelDataset, the "split" arg is true by default
#--- it means (TODO verify) that only a random segment is loaded from each file, per epoch. The default segment is 8192 @ 22050Hz
#--- we use 44100Hz, in addition I doubled it, so total it's 4*[default]
ARGS+=" --segment_size 32768"

[ "$USE_SYNTH_WAVS" = true ]  && ARGS+=" --use_synthesized_wavs"
[ "$USE_SYNTH_WAVS" = true ]  && ARGS+=" --input_mels_dir $SYNTH_MEL_DIR"
[ "$AMP" = true ]             && ARGS+=" --amp"
[ "$FINE_TUNE_DIR" != "" ]    && ARGS+=" --input_mels_dir $FINE_TUNE_DIR"
[ "$FINE_TUNE_DIR" != "" ]    && ARGS+=" --fine_tuning"
[ -n "$FINE_TUNE_LR_FACTOR" ] && ARGS+=" --fine_tune_lr_factor $FINE_TUNE_LR_FACTOR"
[ -n "$EPOCHS_THIS_JOB" ]     && ARGS+=" --epochs_this_job $EPOCHS_THIS_JOB"
[ -n "$SEED" ]                && ARGS+=" --seed $SEED"
[ -n "$GRAD_ACCUMULATION" ]   && ARGS+=" --grad_accumulation $GRAD_ACCUMULATION"
[ "$RESUME" = true ]          && ARGS+=" --resume"
[ -n "$LOG_FILE" ]            && ARGS+=" --log_file $LOG_FILE"
[ -n "$BMARK_EPOCHS_NUM" ]    && ARGS+=" --benchmark_epochs_num $BMARK_EPOCHS_NUM"
[ -n "$MAX_WAV_VALUE" ]       && ARGS+=" --max_wav_value $MAX_WAV_VALUE"

: ${DISTRIBUTED:="-m torch.distributed.launch --nproc_per_node $NUM_GPUS"}
echo "python $DISTRIBUTED train.py $ARGS "$@""
echo "python $DISTRIBUTED train.py $ARGS "$@"" > $OUTPUT_DIR/run_cmd.txt
python $DISTRIBUTED train.py $ARGS "$@"
