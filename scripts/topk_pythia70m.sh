uv run src/Top_AbsoluteK/trainer.py --model_name "EleutherAI/pythia-70m" \
 --model_layer 3 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --steps 30000 

uv run src/Top_AbsoluteK/trainer.py --model_name "EleutherAI/pythia-70m" \
 --model_layer 4 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --steps 30000 