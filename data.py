import copy

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset


def preprocess_data(examples):
    batch_size = len(examples["prompt"])
    processed_prompts = []
    processed_images = []
    for i in range(batch_size):
        prompt_data = examples["prompt"][i]
        image_data = examples["images"][i]
        processed_prompts.append(copy.deepcopy(prompt_data))
        image_index = 0
        for message in prompt_data:
            for content in message["content"]:
                if isinstance(content, dict) and content.get("type") == "image":
                    content["image"] = image_data[image_index]
                    image_index += 1
        processed_images_data, _ = process_vision_info(prompt_data)
        processed_images.append(processed_images_data)
    examples["images"] = processed_images
    examples["prompt"] = processed_prompts
    return examples


class GRPODataset(Dataset):
    def __init__(self, split: str):
        ds = load_dataset("HuggingFaceH4/rlaif-v_formatted")[split]
        self.data = ds.with_transform(preprocess_data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        return {"prompt": sample["prompt"], "images": sample["images"]}
