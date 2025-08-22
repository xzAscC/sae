for k in 50 100 150 200 250 300 350 400 450 500; do
uv run src/Top_AbsoluteK/trainer.py --model_name "google/gemma-2-2b" \
 --model_layer 12 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --save_steps 2000 \
 --steps 10000 \
 --k $k
done