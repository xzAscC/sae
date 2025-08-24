for model_name in "EleutherAI/pythia-70m"; do
    if [ $model_name == "EleutherAI/pythia-70m" ]; then
        ks=(10 20 30 40 50 60 70 80 90 100)
    fi

    for k in ${ks[@]}; do
        uv run src/Top_AbsoluteK/trainer.py \
        --model_name $model_name \
        --model_layer 3 \
        --dataset "monology/pile-uncopyrighted" \
        --sae_name "topk" \
        --batch_size 128 \
        --steps 10000 \
        --log_path "logs/topk_training_k_selection" \
        --save_steps 2000 \
        --k $k
    done
done