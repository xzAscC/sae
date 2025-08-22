uv run src/Top_AbsoluteK/trainer.py --model_name "openai-community/gpt2" \
 --model_layer 8 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --steps 10000

uv run src/Top_AbsoluteK/trainer.py --model_name "google/gemma-2-2b" \
 --model_layer 12 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --steps 10000

uv run src/Top_AbsoluteK/trainer.py --model_name "Qwen/Qwen3-4B-Thinking-2507" \
 --model_layer 20 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --steps 10000
