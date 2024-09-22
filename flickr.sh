#!/bin/bash
set -e
METHOD='FSFH'
bits=(16 32 64 128)

for i in ${bits[*]}; do
  echo "**********Start ${METHOD} algorithm**********"
  CUDA_VISIBLE_DEVICES=0 python train.py --nbit $i \
                                                     --dataset flickr \
                                                     --epochs 140 \
                                                     --dropout 0.1 \
                                                     --mlpdrop 0.1 \

  echo "**********End ${METHOD} algorithm**********"
done

