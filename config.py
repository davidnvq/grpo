from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    batch_size: int = 2
    max_completion_len: int = 256
    num_generations: int = 2
    num_epochs: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    grad_norm: float = 1.0
    epsilon: float = 0.2
    epsilon_high: float = 0.2
    beta: float = 0.04
    temperature: float = 0.9
    top_k: int = 50
    use_fsdp: bool = False
    bf16: bool = False
    fsdp_bf16: bool = False
    gradient_checkpoint: bool = False
    log_steps: int = 1
    save_steps: int = 5
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    push_to_hub: bool = False
    hub_repo_id: str = "YOUR_HUB_REPO_ID"
    hub_private: bool = True
    seed: int = 42
    dtype: str = "bfloat16"
