# Mamba-LLaVA

## Install
First follow the [LLaVA README](./llava.md) create the base environment.

Then install the packages for [Mamba](https://github.com/state-spaces/mamba)

```
pip install causal-conv1d
pip install mamba-ssm
```

## Train

### Pretrain (feature alignment)

Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions we use in the paper [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Pretrain takes around 11 hours for Mamba-2.8B-LLaVA-v1.5 on 4x 3090 (24G).

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](./scripts/mamba/pretrain.sh).

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.

### Visual Instruction Tuning

coming soon ...
