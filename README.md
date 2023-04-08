# Training Command

```PYTHONPATH=../ijsc-ext python train.py --config-path=./train-config/ijsc --config-name=tr_clinc150_val_snips_config_20to6.yaml hydra.run.dir=./output/ijsc/tr_clinc150_val_snips_config_20to6```

# Evaluation Command
```PYTHONPATH=../ijsc-ext python eval.py --config-path=./eval-config/ijsc --config-name=ft_clinc150_downstream_snips_ns20to6.yaml hydra.run.dir=./output/ijsc/ft_clinc150_downstream_snips_ns20to6```

**NOTES for fewshot**
1. Fewshot configuration files have been uploaded. Please stick to configs with 2 negative examples for both 1shot and 2shot for all the datasets.
2. Same commands work for fewshot. Configuration files are required to be changed in the commands relative to hydra.run.dir of your choosing.
3. Few shot data path should be changed in the respective configuration file relative to hydra.run.dir.
4. Models fine-tuned earlier are required to be loaded for fewshot training. **Their paths are also required to be be changed in the fewshot config files**. Refer to the respective few-shot configuration files. 
 
