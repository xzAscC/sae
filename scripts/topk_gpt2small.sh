uv run src/Top_AbsoluteK/trainer.py --model_name "openai-community/gpt2" \
 --model_layer 6 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --steps 30000 

uv run src/Top_AbsoluteK/trainer.py --model_name "openai-community/gpt2" \
 --model_layer 8 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --steps 30000 