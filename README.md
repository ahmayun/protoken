

# Important to update uv 
```
uv self update
```

# important commands 
```
# 16 cores on one node.  
# for arc 
interact -N1 --ntasks-per-node=1 --cpus-per-task=16 -A semcache -p dgx_normal_q -t 6:00:00 --gres=gpu:1  

#a100_normal_q
interact -N1 --ntasks-per-node=1 --cpus-per-task=32 -A semcache -p a100_normal_q -t 6:00:00 --gres=gpu:1

# for h200 
interact -N1 --ntasks-per-node=1 --cpus-per-task=16 -A semcache -p h200_preemptable_q -t 3:00:00 --gres=gpu:1

# for h200_normal_q
interact -N1 --ntasks-per-node=1 --cpus-per-task=16 -A semcache -p h200_normal_q -t 3:00:00 --gres=gpu:1

# for falcon

interact -N1 --gres=gpu:2 -p l40s_normal_q --ntasks-per-node=1 --cpus-per-task=32 -A semcache -t 6:00:00

# for tinkercliff  h200_normal_q
interact -N1 --gres=gpu:1 -p h200_normal_q --ntasks-per-node=1 --cpus-per-task=16 -A semcache -t 6:00:00

interact -N1 --gres=gpu:1 -p h200_preemptable_q --ntasks-per-node=1 --cpus-per-task=32 -A semcache -t 6:00:00

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



