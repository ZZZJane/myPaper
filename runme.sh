#!/bin/bash

MINIDATA=0
 if [ $MINIDATA -eq 1 ]; then
  WORKSPACE="workspace"
  mkdir $WORKSPACE
  TR_SPEECH_DIR="mini_data/train_speech"
  TR_NOISE_DIR="mini_data/train_noise"
  TE_SPEECH_DIR="mini_data/test_speech"
  TE_NOISE_DIR="mini_data/test_noise"
  echo "Using mini data. "
elif [ $MINIDATA -eq 2 ]; then
  WORKSPACE="workspace"
  mkdir $WORKSPACE
  TR_SPEECH_DIR="vol/train_speech"
  TR_NOISE_DIR="vol/train_noise"
  TE_SPEECH_DIR="vol/test_speech"
  TE_NOISE_DIR="vol/test_noise"
  echo "Using part data. "
else
  WORKSPACE="workspace_01"
  mkdir $WORKSPACE
  VOICE_LIB="vol"
  TR_SPEECH_DIR="vol/train"
  TR_NOISE_DIR="vol/train_noise"
  TE_SPEECH_DIR="vol/test"
  TE_NOISE_DIR="vol/test_noise"
  echo "Using all data. "
fi

# Create mixture csv. zxy:get all mixtures
#python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --speech_size=2000 #--magnification=10
#python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --speech_size=200

# Calculate mixture features.
#TR_SNR=0
#TE_SNR=0
#python prepare_data.py calculate_mixture_features --workspace=$VOICE_LIB --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$TR_SNR
#python prepare_data.py calculate_mixture_features --workspace=$VOICE_LIB --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=$TE_SNR

if false;then
for i in 0 5 10 15;do
  echo $i;
  TR_SNR=$i
  TE_SNR=$i
  python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$TR_SNR
  python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=$TE_SNR
done
fi

## Pack features.
N_CONCAT=11
N_HOP=5
#python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP --speech_size=8000
#python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --snr=$TE_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP --speech_size=4000

if false;then
for i in 0 5 10 15;do
  echo $i;
  TR_SNR=$i
  TE_SNR=$i
  python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP
  python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --snr=$TE_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP
done
fi

# Compute scaler.
#python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR

if false;then
for i in 0 5 10 15;do
  echo $i;
  TR_SNR=$i
  python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR
done
fi

# Train.
LEARNING_RATE=1e-4
TR_SNR=0
TE_SNR=0
CUDA_VISIBLE_DEVICES=0 python main_dnn_1to1.py train --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --lr=$LEARNING_RATE
#CUDA_VISIBLE_DEVICES=0 python main_dnn_1to1.py train_noise --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --lr=$LEARNING_RATE

# Plot training stat.
#TR_SNR=15
#python evaluate.py plot_training_stat --workspace=$WORKSPACE --tr_snr=$TR_SNR --bgn_iter=0 --fin_iter=3001 --interval_iter=50

# Inference, enhanced wavs will be created.
ITERATION=3000
TE_SNR=0
#CUDA_VISIBLE_DEVICES=0 python main_dnn_1to1.py inference --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --n_concat=$N_CONCAT --iteration=$ITERATION #--visualize
#CUDA_VISIBLE_DEVICES=0 python main_dnn_1to1.py inference_noise --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --n_concat=$N_CONCAT --iteration=$ITERATION #--visualize

# Calculate PESQ of all enhanced speech.
#python evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$TE_SNR #--cal_type=speech
#python evaluate.py calculate_pesq --workspace="workspace_3" --speech_dir="vol/test" --te_snr=0 #--cal_type=speech

# Calculate PESQ of all enhanced noise.
#python evaluate.py calculate_pesq_noise --workspace=$WORKSPACE --speech_dir=$TE_NOISE_DIR --te_snr=$TE_SNR #--cal_type=noise
#python evaluate.py calculate_pesq_noise --workspace="workspace_3" --speech_dir="vol/test_noise" --te_snr=0 #--cal_type=speech

# Calculate overall stats.
#python evaluate.py get_stats
if false;then
for i in 15;do
  TE_SNR=$i
  echo "test snr="
  echo $i;
#  CUDA_VISIBLE_DEVICES=0 python main_dnn_1to1.py inference --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --n_concat=$N_CONCAT --iteration=$ITERATION #--visualize
  python evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$TE_SNR #--cal_type=speech
  python evaluate.py get_stats
done
fi

