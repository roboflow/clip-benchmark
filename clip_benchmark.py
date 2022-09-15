from dataclasses import dataclass
from time import perf_counter

import clip
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import onnxruntime as ort
from pathlib import Path
import pandas as pd

MODEL_NAME = "ViT-L/14"

@dataclass
class BenchmarkResult:
    times: int
    mean: float
    std: float

    def to_df(self, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame(
            data={
                "times": [self.times],
                "mean (ms)": [self.mean *1000],
                "std (ms)": [self.std * 1000],
                **kwargs,
            }
        )

        return df

    def to_csv(self, filepath: Path, **kwargs):
        df = self.to_df(**kwargs)
        if filepath.exists():
            old_df = pd.read_csv(filepath)
            df = pd.concat([old_df, df])
            # df = df.reset_index()
        df.to_csv(filepath, index=False)



def benchmark_torch(device_id: str, batch_size: int, jit: bool):
    device = torch.device(device_id)
    print(f"Loading model, {jit=}")
    model, _ = clip.load(MODEL_NAME, jit=jit, device=device)
    
    image = torch.randn((batch_size, 3, 224, 224), device=device)
    with torch.no_grad():
        # warmap
        for _ in range(10):
            model.encode_image(image)

        n = 50
        times = []
        torch.cuda.synchronize()
        for _ in tqdm(range(n)):
            start = perf_counter()
            model.encode_image(image).shape
            times.append(perf_counter() - start)
            times_t = torch.as_tensor(times)
    
    times_t = torch.as_tensor(times)

    return BenchmarkResult(times=n, mean=times_t.mean().item(), std=times_t.std().item())

def benchmark_onnx(device_id: str, batch_size: int, quantizate: bool = False):
    providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if device_id == "cuda":
        providers = ["CUDAExecutionProvider"]
        torch.cuda.synchronize()

    model_path = f"./{MODEL_NAME.replace('/', '_')}.quant.onnx" if  quantizate else f"./{MODEL_NAME.replace('/', '_')}.onnx"
    session = ort.InferenceSession(model_path, providers=providers)

    x = torch.randn((batch_size, 3, 224, 224)).numpy()
    # see https://onnxruntime.ai/docs/api/python/api_summary.html#data-on-device
    image = ort.OrtValue.ortvalue_from_numpy(x, device_id, 0)
    output = ort.OrtValue.ortvalue_from_shape_and_type(
        [batch_size, 512], x.dtype, device_id, 0
    )
    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input("image", image)
    io_binding.bind_ortvalue_output("output", output)

    # warmap
    for _ in range(10):
        session.run_with_iobinding(io_binding)

    times = []
    n = 50
    for _ in tqdm(range(n)):
        start = perf_counter()
        session.run_with_iobinding(io_binding)
        times.append(perf_counter() - start)
    times_t = torch.as_tensor(times)

    return BenchmarkResult(times=n, mean=times_t.mean().item(), std=times_t.std().item()
)


if __name__ == "__main__":
    from argparse import ArgumentParser

    # use it like python clip_benchmark.py --device_id="cpu" --runtime="torch_jit"
    parser = ArgumentParser()
    parser.add_argument("--device_id", type=str, default="cuda")
    parser.add_argument("--runtime", type=str, default="torch")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    device_id, runtime, batch_size = args.device_id, args.runtime, args.batch_size
    print(f"Using {device_id=}")

    if "torch" in runtime:
        result = benchmark_torch(device_id, batch_size, jit="jit" in runtime)
    elif "onnx" in runtime:
        result = benchmark_onnx(device_id, batch_size, quantizate = "quant" in runtime)
    else:
        raise ValueError(f"Runtime {runtime} not supported.")
    
    result.to_csv(Path("./result.csv"), device_id=device_id, runtime=runtime, batch_size=batch_size)
