import atexit
import base64
import time
from io import BytesIO
from typing import Optional

import requests
import torch
from PIL import Image
from requests import ConnectionError
from torch import nn
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


def pil_to_base64(pil_image: Image, format="PNG") -> str:
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    base64_string = base64.b64encode(img_bytes).decode("utf-8")
    return base64_string


class VLLMClient:
    def __init__(
        self,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        self.session = requests.Session()
        self.host = host
        self.server_port = server_port
        self.group_port = group_port
        self.check_server(connection_timeout)

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()
        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"server can't be reached at {self.host}:{self.server_port} after {total_timeout}"
                    ) from exc
            else:
                if response.status_code == 200:
                    print("Server is up!")
                    return None
            print(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str] | list[dict],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[str]]:
        url = f"http://{self.host}:{self.server_port}/generate/"
        if isinstance(prompts[0], dict):
            for prompt_dict in prompts:
                image_data = prompt_dict["multi_modal_data"]["image"]
                if isinstance(image_data, list):
                    prompt_dict["multi_modal_data"]["image"] = [
                        pil_to_base64(img) for img in image_data
                    ]
                else:
                    prompt_dict["multi_modal_data"]["image"] = pil_to_base64(image_data)
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
            },
        )
        if response.status_code == 200:
            return response.json()["completion_ids"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_communicator(self):
        url = f"http://{self.host}:{self.server_port}/get_world_size/"
        response = requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        world_size = vllm_world_size + 1
        self.rank = vllm_world_size
        url = f"http://{self.host}:{self.server_port}/init_communicator/"
        response = self.session.post(
            url,
            json={"host": "0.0.0.0", "port": self.group_port, "world_size": world_size},
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        time.sleep(0.1)
        pg = StatelessProcessGroup.create(
            host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device=0)
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"http://{self.host}:{self.server_port}/update_named_param/"
        response = self.session.post(
            url, json={"name": name, "dtype": dtype, "shape": shape}
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

    def update_model_params(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        url = f"http://{self.host}:{self.server_port}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        url = f"http://{self.host}:{self.server_port}/close_communicator/"
        try:
            response = self.session.post(url)
        except ConnectionError:
            pass
        else:
            if response.status_code != 200:
                raise Exception(
                    f"Request failed: {response.status_code}, {response.text}"
                )
