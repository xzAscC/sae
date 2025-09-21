uv run src/trainer.py \
    --model_name "google/gemma-2-2b" \
    --layer 16 \
    --dataset "monology/pile-uncopyrighted" \
    --batch_size 128 \
    --k 230 \
    --sae_name "batchtopk" \
    --training_steps 30000 \
    --checkpoint_path "checkpoints/gemma-2-2b_pile-uncopyrighted_blocks.16.hook_resid_post_16_batchtopk_230_0.0003/sae_20000.safetensors"


uv run src/trainer.py \
    --model_name "Qwen/Qwen3-4B" \
    --layer 20 \
    --dataset "monology/pile-uncopyrighted" \
    --batch_size 128 \
    --k 256 \
    --sae_name "batchtopk" \
    --training_steps 30000 \
    --checkpoint_path "checkpoints/Qwen3-4B_pile-uncopyrighted_blocks.20.hook_resid_post_16_batchtopk_256_0.0003/sae_20000.safetensors"



uv run src/trainer.py \
    --model_name "Qwen/Qwen3-4B" \
    --layer 20 \
    --dataset "monology/pile-uncopyrighted" \
    --batch_size 128 \
    --k 256 \
    --sae_name "batchabsolutek" \
    --training_steps 30000 \
    --checkpoint_path "checkpoints/Qwen3-4B_pile-uncopyrighted_blocks.20.hook_resid_post_16_batchabsolutek_256_0.0003/sae_20000.safetensors"