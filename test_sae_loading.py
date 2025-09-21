#!/usr/bin/env python3
"""
Test script to verify that SAE loading and forward pass work correctly
"""

import sys
import os
sys.path.append('src')

from vis_compare_core_evaluation import load_sae_model
from data import ActivationsStore
import transformer_lens

def test_sae_loading():
    """Test that SAE models can be loaded and used without shape errors."""
    
    print("Testing SAE loading and forward pass...")
    
    # Test paths
    topk_cfg_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_batchtopk_51_0.0003/config_25000.json"
    topk_tensor_path = "checkpoints/pythia-70m_pile-uncopyrighted_blocks.3.hook_resid_post_16_batchtopk_51_0.0003/sae_30000.safetensors"
    
    if not os.path.exists(topk_cfg_path):
        print(f"❌ Config file not found: {topk_cfg_path}")
        return False
    
    if not os.path.exists(topk_tensor_path):
        print(f"❌ Tensor file not found: {topk_tensor_path}")
        return False
    
    try:
        # Load SAE model
        print("Loading TopK SAE...")
        sae, cfg = load_sae_model(topk_cfg_path, topk_tensor_path, "batchtopk")
        print(f"✅ SAE loaded successfully")
        print(f"   Model: {cfg['model_name']}")
        print(f"   Act size: {cfg['act_size']}")
        print(f"   Dict size: {cfg['dict_size']}")
        print(f"   Device: {cfg['device']}")
        print(f"   Dtype: {cfg['dtype']}")
        
        # Load model
        print("Loading transformer model...")
        model = transformer_lens.HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
        print(f"✅ Model loaded successfully")
        print(f"   Model d_model: {model.cfg.d_model}")
        
        # Create activations store
        print("Creating activations store...")
        activations_store = ActivationsStore(model, cfg)
        print(f"✅ Activations store created successfully")
        
        # Test forward pass
        print("Testing SAE forward pass...")
        batch = activations_store.next_batch()
        print(f"   Batch shape: {batch.shape}")
        
        # Forward pass
        x, x_reconstruct, sae_output = sae(batch, return_dict=False)
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Reconstruct shape: {x_reconstruct.shape}")
        print(f"   SAE output keys: {sae_output.keys()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sae_loading()
    if success:
        print("\n🎉 All tests passed! The shape mismatch error should be fixed.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
