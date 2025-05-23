# simple vlm grpo

A simple grpo trainer script for vlms. It's basically a rewrite of TRL's GRPOTrainer but simplified. The idea is to drop some things that work ootb in TRL in exchange for extensibility.

- no accelerate, only torch dist
- supports fsdp
- no weighing rewards
- always scale
- bpo-style loss

it works with qwen2.5-vl but shouldn't be too hard to port it to other archs.

if you want to use it you should:

1. have a look at config.py and update it according to your needs. it should have ~ the same defaults as TRL now
2. update data.py to include whichever data you want to use.

then:

## install

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

---

note: for the following, set the CUDA_VISIBLE_DEVICES for the vllm server and the trainer scripts, similar to TRL's vllm instructions. also, set the --nproc_per_node flag

---

## run vllm server


```bash
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1... uv run vllm_server.py --model "Qwen/Qwen2.5-VL-7B-Instruct"
```

## run train script

```bash
CUDA_VISIBLE_DEVICES=4,5... uv run torchrun --nproc_per_node=4 train.py
```

optionally, you can enable features with flags. e.g.

```bash
CUDA_VISIBLE_DEVICES=4,5.. uv run torchrun --nproc_per_node=4 train.py --use_fsdp
```

## todo:

- impl two-sided clipping: https://github.com/huggingface/trl/commit/05bc43e960396581e458195b8388efe6b82cae1f
