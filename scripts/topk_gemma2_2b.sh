uv run src/Top_AbsoluteK/trainer.py --model_name "google/gemma-2-2b" \
 --model_layer 16 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --steps 30000 

uv run src/Top_AbsoluteK/trainer.py --model_name "google/gemma-2-2b" \
 --model_layer 12 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
