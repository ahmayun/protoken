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

- currently cache is not perfect. It can resume but it will not consider the randomly sampling of data points. even all the configurations are same. 



Different Categories datasets
- [iamketan25/roleplay-instructions-dataset · Datasets at Hugging Face](https://huggingface.co/datasets/iamketan25/roleplay-instructions-dataset)
- [iamtarun/python_code_instructions_18k_alpaca · Datasets at Hugging Face](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)
- [qwedsacf/grade-school-math-instructions · Datasets at Hugging Face](https://huggingface.co/datasets/qwedsacf/grade-school-math-instructions)
- [nlpie/Llama2-MedTuned-Instructions · Datasets at Hugging Face](https://huggingface.co/datasets/nlpie/Llama2-MedTuned-Instructions?row=37)
- [axiong/pmc_llama_instructions · Datasets at Hugging Face](https://huggingface.co/datasets/axiong/pmc_llama_instructions?row=71)
- [fedml/PubMedQA_instruction · Datasets at Hugging Face](https://huggingface.co/datasets/fedml/PubMedQA_instruction?row=49)
- fedml/databricks-dolly-15k-niid
- [llm-wizard/dolly-15k-instruction-alpaca-format · Datasets at Hugging Face](https://huggingface.co/datasets/llm-wizard/dolly-15k-instruction-alpaca-format)
-

