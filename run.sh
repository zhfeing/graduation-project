#!/bin/bash

if [[ $HOSTNAME == "vipa210" ]]; then
    echo "run on vipa210"
    python run.py \
    --gpu="2" --lr=0.01 --epochs=2 \
    --train_v="test" --load_v="test" \
    --regularize=0 --retrain=1 --batch_size=256

elif [[ $HOSTNAME == "vipa-Precision-Tower-7910" ]]; then
    echo "run on vipa-Precision-Tower-7910"
    python run.py \
    --gpu="0" --lr=0.01 --epochs=50 \
    --train_v="googlenet-2.0" --load_v="googlenet-2.0" \
    --regularize=1 --retrain=1 --batch_size=256
else
    echo "unknown host name"
fi

rm model_zoo/model/*epoch*
