import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _infer_img_size(session: ort.InferenceSession) -> int | None:
    shape = session.get_inputs()[0].shape
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        if shape[2] == shape[3]:
            return int(shape[2])
    return None


def _prepare_image(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    img = img.crop((left, top, left + crop, top + crop)).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EfficientNet age ONNX.")
    parser.add_argument("--model", required=True, help="Path to .onnx file.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--size", type=int, default=None, help="Override input size.")
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model))
    img_size = args.size or _infer_img_size(session) or 224
    input_name = session.get_inputs()[0].name
    inputs = {input_name: _prepare_image(Path(args.image), img_size)}
    outputs = session.run(None, inputs)
    if len(outputs) < 2:
        raise RuntimeError("ONNX model did not return mean/log_var outputs.")
    mean, log_var = outputs[0], outputs[1]

    mean_val = float(np.asarray(mean).reshape(-1)[0])
    log_var_val = float(np.asarray(log_var).reshape(-1)[0])
    std_val = float(np.exp(0.5 * log_var_val))
    print(f"mean={mean_val:.4f} log_var={log_var_val:.4f} std={std_val:.4f}")


if __name__ == "__main__":
    main()
