import inspect
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PreTrainedModel, get_cosine_schedule_with_warmup

from config import TrainConfig
from data import GRPODataset
from utils import (
    RepeatSampler,
    accepts_kwarg,
    build_batch_sampler,
    create_reference_model,
    gather,
    gather_object,
    init_distributed,
    init_wandb,
    log_wandb,
    nanmax,
    nanmin,
    parse_args,
    save_checkpoint,
    smart_load,
    sync_fsdp_params_to_vllm,
)
from vllm_client import VLLMClient


def score_completions(
    prompts: list[str],
    completions: list[str],
    completion_ids_list: list[int],
    reward_funcs: list[Callable[[list, list, list], list[float]]],
    device: torch.device,
    cfg: TrainConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    output_reward_func = [
        torch.tensor(
            reward(
                prompts=prompts,
                completions=completions,
                completion_ids=completion_ids_list,
            ),
            dtype=torch.float32,
            device=device,
        )
        for reward in reward_funcs
    ]
    rewards_per_func = torch.stack(output_reward_func, dim=1)
    rewards_per_func = gather(rewards_per_func)
    rewards = rewards_per_func.nansum(dim=1)
    mean_grouped_rewards = rewards.view(-1, cfg.num_generations).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, cfg.num_generations).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
        cfg.num_generations, dim=0
    )
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(
        cfg.num_generations, dim=0
    )
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    rank = dist.get_rank()
    process_slice = slice(rank * len(prompts), (rank + 1) * len(prompts))
    advantages = advantages[process_slice]
    return advantages, rewards, rewards_per_func, std_grouped_rewards


def get_log_probs(
    model: FSDP | DDP | PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    logits_to_keep: int,
    cfg: TrainConfig,
    maybe_cast_to_f32: bool = True,
    **model_kwargs,
) -> Tensor:
    forward_model = model.module if hasattr(model, "module") else model
    forward = (
        forward_model.get_base_model().forward
        if hasattr(forward_model, "get_base_model")
        else forward_model.forward
    )
    if accepts_kwarg(forward, "logits_to_keep"):
        model_kwargs["logits_to_keep"] = logits_to_keep + 1
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, **model_kwargs
    ).logits
    if cfg.bf16 and maybe_cast_to_f32:
        logits = logits.float()
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    logits = logits / cfg.temperature
    index = input_ids
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(
                row_logits,
                dim=-1,
                dtype=torch.bfloat16 if cfg.bf16 and not maybe_cast_to_f32 else None,
            )
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def prepare_inputs(
    batch: list[dict[str, str]],
    policy_model: FSDP | PreTrainedModel,
    processor: AutoProcessor,
    reward_funcs: list[Callable[[list, list, list], list[float]]],
    vllm_client: VLLMClient | None,
    metrics: defaultdict[str, list[float]],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, Tensor], defaultdict[str, list[float]]]:
    prompts = [x["prompt"] for x in batch]
    # print(f"rank {dist.get_rank()} :{prompts}")
    images = [x["images"] for x in batch if "images" in x]
    if len(images) == 0:
        images = None
    if cfg.no_apply_chat_template:
        prompts_text = prompts
    else:
        prompts_text = [
            processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]
    if images is None:
        prompt_inputs = processor(
            text=prompts_text.copy(),
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to(device)
    else:
        prompt_inputs = processor(
            text=prompts_text.copy(),
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to(device)
    prompt_ids, prompt_mask = (
        prompt_inputs["input_ids"],
        prompt_inputs["attention_mask"],
    )
    remaining_prompt_inputs = {
        k: v
        for k, v in prompt_inputs.items()
        if k not in ["input_ids", "attention_mask"]
    }
    update_vllm_client(policy_model, vllm_client, cfg)
    all_images = gather_object(images) if images is not None else None
    all_prompts_text = gather_object(prompts_text)
    if images is not None:
        vllm_prompts = [
            {"multi_modal_data": {"image": image}, "prompt": prompt}
            for prompt, image in zip(
                all_prompts_text[:: cfg.num_generations],
                all_images[:: cfg.num_generations],
            )
        ]
    else:
        vllm_prompts = all_prompts_text[:: cfg.num_generations]
    rank = dist.get_rank()
    if rank == 0:
        completion_ids = vllm_client.generate(
            prompts=vllm_prompts,
            n=cfg.num_generations,
            max_tokens=cfg.max_completion_len,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )
    else:
        completion_ids = [None] * len(all_prompts_text)
    dist.broadcast_object_list(completion_ids, src=0)
    process_slice = slice(rank * len(prompts), (rank + 1) * len(prompts))
    completion_ids = completion_ids[process_slice]
    completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
    pad_token_id = (
        processor.tokenizer.pad_token_id
        if images is not None
        else processor.pad_token_id
    )
    eos_token_id = (
        processor.tokenizer.eos_token_id
        if images is not None
        else processor.eos_token_id
    )
    completion_ids = torch.nn.utils.rnn.pad_sequence(
        completion_ids, batch_first=True, padding_value=pad_token_id
    ).to(device)
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full(
        (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
    )
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
        is_eos.size(0), -1
    )
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    completion_texts = processor.batch_decode(completion_ids, skip_special_tokens=True)
    completion_ids_list = [
        [id.item() for id, m in zip(row, mask_row) if m]
        for row, mask_row in zip(completion_ids, completion_mask)
    ]
    advantages, rewards, rewards_per_func, std_grouped_rewards = score_completions(
        prompts, completion_texts, completion_ids_list, reward_funcs, device, cfg
    )
    metrics["num_tokens"] = [
        gather(attention_mask.sum()).sum().item()
        + (metrics["num_tokens"][0] if metrics["num_tokens"] else 0)
    ]
    agg_completion_mask = gather_object((completion_mask.sum(1)).tolist())
    metrics["completions/mean_length"].append(
        sum(agg_completion_mask) / len(agg_completion_mask)
    )
    metrics["completions/min_length"].append(min(agg_completion_mask))
    metrics["completions/max_length"].append(max(agg_completion_mask))
    for i, reward_func in enumerate(reward_funcs):
        mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
        metrics[f"rewards/{reward_func.__name__}"].append(mean_rewards)
    metrics["reward"].append(rewards.mean().item())
    metrics["reward_std"].append(std_grouped_rewards.mean().item())
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": advantages,
        **remaining_prompt_inputs,
    }, metrics


def compute_loss(
    policy_model: FSDP | DDP,
    ref_model: FSDP | PreTrainedModel | None,
    inputs: dict[str, Tensor],
    metrics: defaultdict[str, list[float]],
    cfg: TrainConfig,
) -> tuple[Tensor, defaultdict[str, list[float]]]:
    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = (
        inputs["completion_ids"],
        inputs["completion_mask"],
    )
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    model_kwarg_keys = (
        inspect.signature(policy_model.module.forward).parameters.keys()
        if not hasattr(policy_model.module, "get_base_model")
        else inspect.signature(
            policy_model.module.get_base_model().forward
        ).parameters.keys()
    )
    remaining_kwargs = {k: inputs[k] for k in model_kwarg_keys if k in inputs}
    per_token_logps = get_log_probs(
        policy_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        cfg,
        **remaining_kwargs,
    )
    with torch.no_grad():
        if ref_model is None:
            ctxt = (
                policy_model.module.disable_adapter()
                if cfg.use_peft
                else policy_model.disable_adapter()
            )
            with ctxt:
                ref_per_token_logps = get_log_probs(
                    policy_model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    cfg,
                    **remaining_kwargs,
                )
        else:
            ref_per_token_logps = get_log_probs(
                ref_model,
                input_ids,
                attention_mask,
                logits_to_keep,
                cfg,
                maybe_cast_to_f32=False if cfg.fsdp_bf16 and cfg.use_fsdp else True,
                **remaining_kwargs,
            )
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps)
        - 1
    )
    advantages = inputs["advantages"]
    old_per_token_logps = per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_per_token_logps)
    coef_2 = torch.clamp(coef_1, 1 - cfg.epsilon, 1 + cfg.epsilon_high)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    per_token_loss = per_token_loss + cfg.beta * per_token_kl
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(
        min=1.0
    )
    metrics["kl"].append(
        gather((per_token_kl * completion_mask).sum() / completion_mask.sum())
        .nanmean()
        .item()
    )
    is_low_clipped = (coef_1 < 1 - cfg.epsilon) & (advantages.unsqueeze(1) < 0)
    is_high_clipped = (coef_1 > 1 + cfg.epsilon_high) & (advantages.unsqueeze(1) > 0)
    is_region_clipped = is_low_clipped | is_high_clipped
    low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
    high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
    clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()
    gathered_low_clip = gather(low_clip)
    metrics["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
    metrics["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
    gathered_high_clip = gather(high_clip)
    metrics["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
    metrics["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
    gathered_clip_ratio = gather(clip_ratio)
    metrics["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
    return loss, metrics


def update_vllm_client(
    model: FSDP | PreTrainedModel, vllm_client: VLLMClient | None, cfg: TrainConfig
) -> None:
    rank = dist.get_rank()
    if cfg.use_peft:
        if cfg.use_fsdp:
            sync_fsdp_params_to_vllm(model, vllm_client, peft=True)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                model.merge_adapter()
            for name, param in model.named_parameters():
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if model.prefix in name:
                    continue
                if "original_module" in name:
                    continue
                name = name.replace("modules_to_save.default.", "")
                if rank == 0:
                    vllm_client.update_named_param(name, param.data)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                model.unmerge_adapter()
    else:
        if cfg.use_fsdp:
            sync_fsdp_params_to_vllm(model, vllm_client)
        else:
            if rank == 0:
                for name, param in model.named_parameters():
                    vllm_client.update_named_param(name, param.data)
    if rank == 0:
        vllm_client.reset_prefix_cache()


def init_dataloader(split: str, cfg: TrainConfig) -> DataLoader:
    dataset = GRPODataset(cfg.dataset_id, split, cfg.extra_columns)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    per_dev = cfg.batch_size
    gen_per = cfg.num_generations
    sampler = RepeatSampler(
        data_source=dataset,
        mini_repeat_count=gen_per,
        batch_size=(world_size * per_dev) // gen_per,
        repeat_count=1,
        shuffle=True,
        seed=cfg.seed,
    )
    batch_sampler = build_batch_sampler(
        sampler=sampler,
        batch_size=cfg.batch_size,
        num_replicas=world_size,
        rank=rank,
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=cfg.collate_fn,
        num_workers=0,
        pin_memory=True,
    )


def reward_len(completions: list[str], **kwargs) -> list[float]:
    return [-abs(20 - len(completion)) for completion in completions]


def init_models(
    cfg: TrainConfig, local_rank: int, device: torch.device
) -> tuple[FSDP | DDP, FSDP | DDP | None, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(cfg.model_id, padding_side="left")
    if cfg.use_peft:
        policy_model_unwrapped = smart_load(
            cfg.model_id, use_cache=cfg.use_cache, torch_dtype=cfg.dtype
        )
        lora_config = LoraConfig(
            lora_alpha=64,
            lora_dropout=0.05,
            r=32,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        policy_model_unwrapped = get_peft_model(policy_model_unwrapped, lora_config)
        if cfg.use_fsdp and cfg.dtype == torch.bfloat16:
            policy_model_unwrapped.to(torch.bfloat16)
        policy_model_unwrapped.print_trainable_parameters()
        if cfg.gradient_checkpoint:
            if cfg.use_fsdp:
                policy_model_unwrapped.base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            else:
                policy_model_unwrapped.base_model.gradient_checkpointing_enable()
    else:
        policy_model_unwrapped = smart_load(
            cfg.model_id, use_cache=cfg.use_cache, torch_dtype=cfg.dtype
        )
        if cfg.gradient_checkpoint:
            if cfg.use_fsdp:
                policy_model_unwrapped.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            else:
                policy_model_unwrapped.gradient_checkpointing_enable()
    if cfg.gradient_checkpoint:
        policy_model_unwrapped.enable_input_require_grads()
    if cfg.use_fsdp:
        mixed_precision = (
            MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                keep_low_precision_grads=False,
                cast_forward_inputs=False,
                cast_root_forward_inputs=True,
                _module_classes_to_ignore=(nn.modules.batchnorm._BatchNorm,),
            )
            if cfg.fsdp_bf16
            else None
        )
        policy_model = FSDP(
            policy_model_unwrapped,
            device_id=local_rank,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            mixed_precision=mixed_precision,
            sync_module_states=True,
        )
    else:
        policy_model_unwrapped.to(device)
        policy_model = DDP(
            policy_model_unwrapped,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True if cfg.gradient_checkpoint is False else False,
        )
    policy_model.train()
    if cfg.use_fsdp:
        ref_model_unwrapped = smart_load(
            cfg.model_id, use_cache=cfg.use_cache, torch_dtype=cfg.dtype
        )
        ref_model = FSDP(
            ref_model_unwrapped,
            device_id=local_rank,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            mixed_precision=mixed_precision,
            sync_module_states=True,
        )
        ref_model.eval()
    elif cfg.use_peft:
        ref_model = None
    else:
        ref_model_copy = create_reference_model(policy_model_unwrapped)
        ref_model = ref_model_copy.to(device)
        del ref_model_copy
    return policy_model, ref_model, processor


def train(cfg: TrainConfig, local_rank: int, device: torch.device) -> None:
    rank = dist.get_rank()
    metrics = defaultdict(list)
    if cfg.use_wandb and rank == 0:
        init_wandb()
    reward_funcs = [reward_len]
    policy_model, ref_model, processor = init_models(cfg, local_rank, device)
    dataloader = init_dataloader("train", cfg)
    optimizer = AdamW(
        [p for _, p in policy_model.named_parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    num_training_steps = cfg.num_epochs * len(dataloader)
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    vllm_client = VLLMClient(connection_timeout=120.0) if rank == 0 else None
    if rank == 0:
        vllm_client.init_communicator()
    dist.barrier()
    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            policy_model.train()
            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if cfg.bf16
                else nullcontext()
            ):
                inputs, metrics = prepare_inputs(
                    batch,
                    policy_model if cfg.use_fsdp else policy_model.module,
                    processor,
                    reward_funcs,
                    vllm_client,
                    metrics,
                    cfg,
                    device,
                )
                loss, metrics = compute_loss(
                    policy_model, ref_model, inputs, metrics, cfg
                )
            loss.backward()
            metrics["loss"].append(round(gather(loss).mean().item(), 4))
            if cfg.use_fsdp:
                grad_norm_to_log = torch.as_tensor(
                    policy_model.clip_grad_norm_(cfg.grad_norm), device=device
                )
            else:
                grad_norm_to_log = torch.as_tensor(
                    clip_grad_norm_(policy_model.parameters(), cfg.grad_norm),
                    device=device,
                )
            metrics["grad_norm"].append(gather(grad_norm_to_log).mean().item())
            metrics["learning_rate"].append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % cfg.log_steps == 0 and rank == 0:
                metrics_str = " | ".join(f"{k}: {v[-1]}" for k, v in metrics.items())
                print(f"epoch {epoch} | step: {step + 1} | {metrics_str}")
                if cfg.use_wandb:
                    log_wandb(metrics)
            if (step + 1) % cfg.save_steps == 0 or (step + 1) == len(dataloader):
                save_checkpoint(
                    model=policy_model,
                    processor=processor,
                    push_to_hub=cfg.push_to_hub,
                    hub_repo_id=cfg.hub_repo_id,
                    hub_private=cfg.hub_private,
                    commit_msg=f"checkpoint at step {step + 1}"
                    if (step + 1) % cfg.save_steps == 0
                    else "final checkpoint",
                )


if __name__ == "__main__":
    local_rank, device = init_distributed()
    cfg = parse_args()
    train(cfg, local_rank, device)
