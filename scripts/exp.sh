#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

run_base(){
    root_path="$1"
    data_path="$2"
    model="$3"
    data="$4"
    exp_notes="$5"
    seq_len="$6"
    label_len="$7"
    pred_len="$8"
    factor="$9"
    enc_in="${10}"
    dec_in="${11}"
    c_out="${12}"
    
    python -u run.py \
        --is_training 1 \
        --root_path "$root_path" \
        --data_path "$data_path" \
        --model_id "$data"_"$seq_len"_"$pred_len"_"$exp_notes" \
        --model "$model" \
        --data "$data" \
        --features MS \
        --seq_len "$seq_len" \
        --label_len "$label_len" \
        --pred_len "$pred_len" \
        --e_layers 2 \
        --d_layers 1 \
        --factor "$factor" \
        --enc_in "$enc_in" \
        --dec_in "$dec_in" \
        --c_out "$c_out" \
        --des 'Exp' \
        --itr 3 \
        --patience 3    \
        # --use_multi_gpu \
}

run_PV(){
    root_path="../dataset/PV/"
    data_path="$1"
    model="$2" #NOTE
    data="$3"
    exp_notes="$4" #NOTE
    seq_len="$5" #NOTE
    label_len=$(($seq_len/2))
    pred_len="$6" #NOTE
    factor="$7" #NOTE
    enc_in=13
    dec_in=13
    c_out=1

    run_base "$root_path" "$data_path" "$model" "$data" "$exp_notes" \
        "$seq_len" "$label_len" "$pred_len" "$factor" "$enc_in" "$dec_in" "$c_out"
}


files=$(ls)

run_batch(){
    exp_notes=""
#    our_model="InParformer"
    models=("Informer" "FEDformer" "Autoformer")
    for data_path in $files;
    do
      for model in ${models[*]};
      do
          run_PV "$data_path" "$model" "$data" "$exp_notes" 96 96 3
          run_PV "$data_path" "$model" "$data" "$exp_notes" 96 192 3
          run_PV "$data_path" "$model" "$data" "$exp_notes" 96 336 3
          run_PV "$data_path" "$model" "$data" "$exp_notes" 96 720 3

      done
    done
}

run_batch