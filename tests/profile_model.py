import pandas as pd
from peft.tuners.lora import LoraLayer
from src.provenance.fl_prov import get_all_layers
# Assuming these functions are defined in your project
from src.fl.model import get_model_and_tokenizer
from src.fl.config import get_default_config
from src.run_provenance import MODEL2LayerConfig

# --- 1. Model Loading ---
config = get_default_config()
# config['model_config']['model_name'] = "meta-llama/Llama-3.2-1B-Instruct"
config['model_config']['model_name'] = "Qwen/Qwen2.5-0.5B-Instruct"
config['use_lora'] = False


model, tokenizer = get_model_and_tokenizer(config)

# --- 2. Layer-based Profiling Logic ---
layer_data = []

# Iterate over all named modules (layers) in the model
for name, module in model.named_modules():
    print(f'Larer: {name}, Type: {module.__class__.__name__}')
    num_params = 0
    trainable_params = 0
    
    # Iterate over parameters directly attached to this module (not its children)
    for param in module.parameters(recurse=False):
        num_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    # Only include layers that have parameters directly
    if num_params > 0:
        layer_data.append({
            "Layer Name": name,
            "Layer Type": module.__class__.__name__,
            # UPDATED: Check for 'lora' in the layer's name. This is more reliable
            # across different model architectures and PEFT versions.
            "Is LoRA Modified?": 'lora' in name,
            "Total Params": num_params,
            "Trainable Params": trainable_params,
        })

# Create a pandas DataFrame for pretty printing
df = pd.DataFrame(layer_data)

# --- 3. Displaying the Report ---
print("\n" + "="*80)
print("Model Layer Profile")
print("="*80)

# --- Summary Statistics ---
total_params = df['Total Params'].sum()
trainable_params = df['Trainable Params'].sum()
lora_params = trainable_params

print(f"\nTotal Parameters:      {total_params:,}")
print(f"Trainable Parameters:  {trainable_params:,} ({trainable_params / total_params:.4%})")
print(f"Whereof LoRA Params: {lora_params:,} (100% of trainable)")
print("\n" + "-"*80)

# --- Detailed Layer Breakdown ---
print("Detailed Layer Breakdown:")
# Adjust max_rows if your model has more LoRA layers than this
pd.set_option('display.max_rows', 10000) 

print("\n--- Sample of Frozen Base Model Layers ---")
print(df[df['Trainable Params'] == 0].head(10).to_string())

print("\n\n--- All Trainable (LoRA-injected) Layers ---")
print(df[df['Trainable Params'] > 0].to_string())


if config['use_lora']: 
    all_layers = get_all_layers(model, MODEL2LayerConfig['lora'])
else:
    all_layers = get_all_layers(model, MODEL2LayerConfig['standard'])

print('=========== Provenance Layers =====')
for d in all_layers:
    print(d['name'])
