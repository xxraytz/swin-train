
First of all install libs following this README: https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md. 

Also you should install:
- bitsandbytes
- triton==2.0.0

To run training of swin-transformer use this command:

```
TOKEN=<TOKEN> OPTIMIZER=AdamW LINEAR_TYPE=torch python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 swin-train.py --cfg configs/swin_tiny_c24_patch4_window8_256.yaml
```

- Use LINEAR_TYPE=switchback instead of torch if you need switchback.
- Add your hugging-face token in [TOKEN] place to download imagenet dataset.
- If you want to change dataset look config.py and don't forget about dataset preprocessing.
