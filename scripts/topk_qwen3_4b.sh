uv run src/Top_AbsoluteK/trainer.py --model_name "Qwen/Qwen3-4B-Thinking-2507" \
 --model_layer 18 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --steps 30000 

uv run src/Top_AbsoluteK/trainer.py --model_name "Qwen/Qwen3-4B-Thinking-2507" \
 --model_layer 20 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
