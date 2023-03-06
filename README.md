# Training Command

```PYTHONPATH=../ijsc-ext python train.py --config-path=./train-config/ijsc --config-name=tr_clinc150_val_snips_config_20to6.yaml hydra.run.dir=./output/ijsc/tr_clinc150_val_snips_config_20to6```

# Evaluation Command
```PYTHONPATH=../ijsc-ext python eval.py --config-path=./eval-config/ijsc --config-name=ft_clinc150_downstream_snips_ns20to6.yaml hydra.run.dir=./output/ijsc/ft_clinc150_downstream_snips_ns20to6```
