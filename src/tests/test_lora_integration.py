#!/usr/bin/env python3
import torch.nn as nn
from src.config.base_config import ConfigManager
from src.utils.model import get_model_and_tokenizer
from src.utils.model import ModelUtils
from src.provenance.fl_prov import get_all_layers

def has_trainable_params(m: nn.Module, *, recurse=False) -> bool:
    return any(p.requires_grad for p in m.parameters(recurse=recurse))

def count_params(m: nn.Module, *, trainable_only=False, recurse=False) -> int:
    it = (p for p in m.parameters(recurse=recurse))
    if trainable_only:
        it = (p for p in it if p.requires_grad)
    return sum(p.numel() for p in it)

def test_lora_integration():
    print("Testing LoRA Integration...")
    
    # Test 1: Load config and verify LoRA config exists
    print("\n1. Testing configuration loading...")
    config, exp_key = ConfigManager.load_config_with_corresponding_key()
    
    assert "lora_config" in config, "LoRA config missing"
    print(f"✓ LoRA config found: {config['lora_config']}")
    
    # Test 2: Test model loading without LoRA (default)
    print("\n2. Testing model loading without LoRA...")
    model, tokenizer = get_model_and_tokenizer(config)
    print(f"✓ Model loaded successfully: {type(model).__name__}")
    # check if model has peft_config attribute
    assert not hasattr(model, 'peft_config'), "Model should not have peft_config attribute"
    
    # Test 3: Test parameter extraction without LoRA
    print("\n3. Testing parameter extraction without LoRA...")
    params = ModelUtils.get_parameters(model)
    print(f"✓ Parameters extracted: {len(params)} tensors")
    
    # Test 4: Test model loading with LoRA enabled
    print("\n4. Testing model loading with LoRA enabled...")
    config["lora_config"]["use_lora"] = True
    lora_model, lora_tokenizer = get_model_and_tokenizer(config)
    
    print(lora_model)
    for name, module in lora_model.named_modules():
        print(f"Module: {name} | Type: {type(module).__name__}")

    print(" ---------------")
                
    _ = input("Press Enter to continue...")


    print(f"✓ LoRA model loaded successfully: {type(lora_model).__name__}")
    assert hasattr(lora_model, 'peft_config'), "LoRA model should have peft_config attribute"
     
    # Test 5: Test LoRA parameter extraction
    print("\n5. Testing LoRA parameter extraction...")
    lora_params = ModelUtils.get_parameters(lora_model)
    print(f"✓ LoRA parameters extracted: {len(lora_params)} tensors")
    print(f"✓ Parameter count difference: {len(params)} -> {len(lora_params)}")
    
    # Test 6: Test parameter setting
    print("\n6. Testing parameter setting...")
    ModelUtils.set_parameters(lora_model, lora_params)
    print("✓ LoRA parameters set successfully")
    
    print("\n✅ All LoRA integration tests passed!")
    print(f"Experiment key format: {exp_key}")

if __name__ == "__main__":
    test_lora_integration()
