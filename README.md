pip3 install torch torchvision torchaudio 

pip install packaging

pip install ninja

pip install flash-attn --no-build-isolation

pip install scikit-learn evaluate diskcache transformers peft flwr-datasets trl flwr[simulation] hydra-core


# important commands 
```
# 16 cores on one node.  
interact -N1 --ntasks-per-node=1 --cpus-per-task=16 -A semcache -p dgx_normal_q -t 6:00:00 --gres=gpu:1  
```

