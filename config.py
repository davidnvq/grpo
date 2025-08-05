import copy
from dataclasses import dataclass
from typing import Callable

from qwen_vl_utils import process_vision_info


def collate_fn(batch):
    processed_samples = []
    for sample in batch:
        prompt_data = sample["prompt"]
        processed_prompt = copy.deepcopy(prompt_data)
        processed_images = []
        if "images" in sample:
            image_data = sample["images"]
            image_index = 0
            for message in processed_prompt:
                for content in message["content"]:
                    if isinstance(content, dict) and content.get("type") == "image":
                        content["image"] = image_data[image_index]
                        image_index += 1
            processed_images, *_ = process_vision_info(processed_prompt)
        processed_sample = {"prompt": processed_prompt, "images": processed_images}
        for key, value in sample.items():
            if key not in ["prompt", "images"]:
                processed_sample[key] = value
        processed_samples.append(processed_sample)
    return processed_samples


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    dataset_id: str = "HuggingFaceH4/rlaif-v_formatted"
    collate_fn: Callable[[list[dict]], list[dict]] | None = None
    no_apply_chat_template: bool = False
    extra_columns: list[str] | None = None
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
    use_peft: bool = True
    use_fsdp: bool = False
    bf16: bool = True
    fsdp_bf16: bool = True
    gradient_checkpoint: bool = True
    log_steps: int = 1
    save_steps: int = 5
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    push_to_hub: bool = False
    hub_repo_id: str = "YOUR_HUB_REPO_ID"
    hub_private: bool = True
    seed: int = 42
    dtype: str = "bfloat16"
    use_cache: bool = False
