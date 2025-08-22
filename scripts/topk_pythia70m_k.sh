for k in 10 20 30 40 50 60 70 80 90 100; do
uv run src/Top_AbsoluteK/trainer.py --model_name "EleutherAI/pythia-70m" \
 --model_layer 12 \
 --dataset "pyvene/axbench-concept16k_v2" \
 --sae_name "topk" \
 --save_steps 2000 \
 --steps 10000 \
 --k $k
done