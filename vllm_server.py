import argparse
import base64
import os
from argparse import Namespace
from contextlib import asynccontextmanager
from io import BytesIO
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Sequence

import torch
import uvicorn
from fastapi import FastAPI, Request
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.sampling_params import GuidedDecodingParams
from vllm.utils import get_open_port

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def base64_to_pil(base64_string: str) -> Image:
    img_bytes = base64.b64decode(base64_string)
    buffered = BytesIO(img_bytes)
    pil_image = Image.open(buffered)
    return pil_image


class WeightSyncWorkerExtension:
    pynccl_comm = None
    client_rank = None

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        if self.pynccl_comm is not None:
            raise RuntimeError(
                "Weight update group already initialized. Call close_communicator first."
            )
        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1

    def update_named_param(
        self, name: str, dtype: torch.dtype, shape: Sequence[int]
    ) -> None:
        if self.pynccl_comm is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )
        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


def llm_worker(
    args: Namespace, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    llm = LLM(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        enable_prefix_caching=args.enable_prefix_caching,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.image_limit_mm_per_prompt}
        if args.image_limit_mm_per_prompt
        else None,
        worker_extension_cls=f"{__name__}.WeightSyncWorkerExtension",
    )
    connection.send({"status": "ready"})
    while True:
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            llm.collective_rpc(method="close_communicator")
            break
        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
            method = getattr(llm, method_name)
            result = method(*args, **kwargs)
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break


def chunk_list(lst: list, n: int) -> list[list]:
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--enable_prefix_caching", type=bool, default=None)
    parser.add_argument("--enforce_eager", type=bool, default=None)
    parser.add_argument("--kv_cache_dtype", type=str, default="auto")
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--image_limit_mm_per_prompt", type=int, default=None)
    args = parser.parse_args()
    master_port = get_open_port()
    connections = []
    processes = []
    for data_parallel_rank in range(args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(
            target=llm_worker,
            args=(args, data_parallel_rank, master_port, child_connection),
        )
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ready_connections = set()
        while len(ready_connections) < args.data_parallel_size:
            for connection in connections:
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get("status") == "ready":
                    ready_connections.add(connection)
        yield
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        return {"world_size": args.tensor_parallel_size * args.data_parallel_size}

    @app.post("/generate/")
    async def generate(request: Request):
        body = await request.json()
        prompts = body.get("prompts")
        if isinstance(prompts[0], dict):
            for prompt_dict in prompts:
                image_data = prompt_dict["multi_modal_data"]["image"]
                if isinstance(image_data, list):
                    prompt_dict["multi_modal_data"]["image"] = [
                        base64_to_pil(img) for img in image_data
                    ]
                else:
                    prompt_dict["multi_modal_data"]["image"] = base64_to_pil(image_data)
        sampling_params = SamplingParams(
            n=body.get("n", 1),
            repetition_penalty=body.get("repetition_penalty", 1.0),
            temperature=body.get("temperature", 1.0),
            top_p=body.get("top_p", 1.0),
            top_k=body.get("top_k", -1),
            min_p=body.get("min_p", 0.0),
            max_tokens=body.get("max_tokens", 16),
            guided_decoding=GuidedDecodingParams(
                backend="outlines", regex=body["guided_decoding_regex"]
            )
            if body.get("guided_decoding_regex")
            else None,
        )
        chunked_prompts = chunk_list(prompts, args.data_parallel_size)
        for connection, prompts in zip(connections, chunked_prompts):
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})
        all_outputs = [connection.recv() for connection in connections]
        all_outputs = [
            output for output, prompts in zip(all_outputs, chunked_prompts) if prompts
        ]
        all_outputs = list(chain.from_iterable(all_outputs))
        return {
            "completion_ids": [
                list(output.token_ids)
                for outputs in all_outputs
                for output in outputs.outputs
            ]
        }

    @app.post("/init_communicator/")
    async def init_communicator(request: Request):
        body = await request.json()
        world_size = args.tensor_parallel_size * args.data_parallel_size + 1
        kwargs = {
            "method": "init_communicator",
            "args": (body.get("host"), body.get("port"), world_size),
        }
        for connection in connections:
            connection.send(
                {
                    "type": "fire_and_forget",
                    "method": "collective_rpc",
                    "kwargs": kwargs,
                }
            )
        return {"message": "Request received, initializing communicator"}

    @app.post("/update_named_param/")
    async def update_named_param(request: Request):
        body = await request.json()
        name = body.get("name")
        dtype_str = body.get("dtype")
        shape = body.get("shape")
        dtype = torch.__getattribute__(dtype_str.split(".")[-1])
        kwargs = {"method": "update_named_param", "args": (name, dtype, tuple(shape))}
        for connection in connections:
            connection.send(
                {
                    "type": "fire_and_forget",
                    "method": "collective_rpc",
                    "kwargs": kwargs,
                }
            )
        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        for connection in connections:
            connection.send({"type": "call", "method": "reset_prefix_cache"})
        all_outputs = [connection.recv() for connection in connections]
        success = all(output for output in all_outputs)
        return {
            "message": "Request received, resetting prefix cache status: "
            + str(success)
        }

    @app.post("/close_communicator/")
    async def close_communicator():
        kwargs = {"method": "close_communicator"}
        for connection in connections:
            connection.send(
                {
                    "type": "fire_and_forget",
                    "method": "collective_rpc",
                    "kwargs": kwargs,
                }
            )
        return {"message": "Request received, closing communicator"}

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
