import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKEND = os.environ.get("TF_DEPLOY_BACKEND", "onnx").lower()
MODEL_CACHE = {}


MODEL_SPECS = {
    "seg_tf": {
        "engine": "seg_tf.engine",
        "onnx": "end2end_tunnelfaceseg.onnx",
        "input": "input",
        "shape": (1, 3, 512, 512),
    },
    "cla_posui": {
        "engine": "cla_posui.engine",
        "onnx": "classification_posui.onnx",
        "input": "input_1",
        "shape": (1, 299, 299, 3),
    },
    "seg_water": {
        "engine": "seg_water.engine",
        "onnx": "end2end_waterseg.onnx",
        "input": "input",
        "shape": (1, 3, 512, 512),
    },
    "cla_ytlx": {
        "engine": "cla_ytlx.engine",
        "onnx": "end2end_YTLX.onnx",
        "input": "input",
        "shape": (1, 3, 224, 224),
    },
    "cla_fhcd": {
        "engine": "cla_fhcd.engine",
        "onnx": "end2end_FHCD.onnx",
        "input": "input",
        "shape": (1, 3, 224, 224),
    },
    "crackseg_segformer": {
        "engine": "crackseg_segformer.engine",
        "onnx": "crackseg_segformer.onnx",
        "input": "input",
        "shape": (1, 3, 512, 512),
    },
}

ENGINE_TO_MODEL = {
    spec["engine"]: model_name for model_name, spec in MODEL_SPECS.items()
}
ONNX_TO_MODEL = {
    spec["onnx"]: model_name for model_name, spec in MODEL_SPECS.items()
}


def preprocess(image):
    image = image.convert("RGB")
    mean = np.array([0.485, 0.456, 0.406]).astype("float32")
    stddev = np.array([0.229, 0.224, 0.225]).astype("float32")
    data = (np.asarray(image).astype("float32") / 255.0 - mean) / stddev
    return np.moveaxis(data, 2, 0)


def postprocess(data):
    num_classes = 21
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([palette * i % 255 for i in range(num_classes)]).astype("uint8")
    img = Image.fromarray(data.astype("uint8"), mode="P")
    img.putpalette(colors)
    return img


@dataclass
class RuntimeModel:
    backend: str
    model_name: str
    model_path: Path
    session: object
    input_name: str


def _model_name_from_path(model_path):
    name = Path(model_path).name
    if name in ENGINE_TO_MODEL:
        return ENGINE_TO_MODEL[name]
    if name in ONNX_TO_MODEL:
        return ONNX_TO_MODEL[name]
    stem = Path(model_path).stem
    if stem in MODEL_SPECS:
        return stem
    raise ValueError(f"Unknown model path: {model_path}")


def _resolve_model_path(requested_path, backend, model_name):
    requested_path = Path(requested_path)
    if backend == "tensorrt":
        if requested_path.suffix == ".engine":
            return requested_path
        return PROJECT_ROOT / "model" / MODEL_SPECS[model_name]["engine"]
    if backend == "onnx":
        if requested_path.suffix == ".onnx":
            return requested_path
        return PROJECT_ROOT / "ori_model" / MODEL_SPECS[model_name]["onnx"]
    raise ValueError("TF_DEPLOY_BACKEND must be 'onnx' or 'tensorrt'")


def _onnx_providers():
    providers = os.environ.get("TF_DEPLOY_ONNX_PROVIDERS")
    if providers:
        return [provider.strip() for provider in providers.split(",") if provider.strip()]
    return ["CPUExecutionProvider"]


def _cache_key(backend, model_path):
    provider_key = tuple(_onnx_providers()) if backend == "onnx" else ()
    log_level = os.environ.get("TF_DEPLOY_ONNX_LOG_SEVERITY", "3")
    return backend, str(Path(model_path).resolve()), provider_key, log_level


def _load_onnx_model(model_path, model_name):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is not installed. Run `conda env create -f environment.yml` "
            "and activate the tf_deploy environment."
        ) from exc

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = int(
        os.environ.get("TF_DEPLOY_ONNX_LOG_SEVERITY", "3")
    )
    providers = _onnx_providers()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )
    expected_input = MODEL_SPECS[model_name]["input"]
    input_names = [model_input.name for model_input in session.get_inputs()]
    input_name = expected_input if expected_input in input_names else input_names[0]
    logging.info(f"Reading ONNX model from file {model_path}")
    logging.info(f"Using ONNX Runtime providers: {session.get_providers()}")
    return RuntimeModel("onnx", model_name, model_path, session, input_name)


def _load_tensorrt_model(model_path, model_name):
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is not installed. Install TensorRT on the NVIDIA deployment "
            "machine or use TF_DEPLOY_BACKEND=onnx."
        ) from exc

    if not model_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {model_path}")

    logger = trt.Logger()
    logging.info(f"Reading TensorRT engine from file {model_path}")
    with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return RuntimeModel(
        "tensorrt",
        model_name,
        model_path,
        engine,
        MODEL_SPECS[model_name]["input"],
    )


def load_model(model_path, backend: Optional[str] = None):
    backend = (backend or os.environ.get("TF_DEPLOY_BACKEND", DEFAULT_BACKEND)).lower()
    model_name = _model_name_from_path(model_path)
    resolved_path = _resolve_model_path(model_path, backend, model_name)
    key = _cache_key(backend, resolved_path)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    if backend == "onnx":
        model = _load_onnx_model(resolved_path, model_name)
    elif backend == "tensorrt":
        model = _load_tensorrt_model(resolved_path, model_name)
    else:
        raise ValueError("TF_DEPLOY_BACKEND must be 'onnx' or 'tensorrt'")

    MODEL_CACHE[key] = model
    return model


def clear_model_cache():
    MODEL_CACHE.clear()


def load_engine(engine_file_path):
    return load_model(engine_file_path)


def _run_onnx(model, input_image):
    input_image = np.ascontiguousarray(input_image.astype("float32"))
    outputs = model.session.run(None, {model.input_name: input_image})
    if not outputs:
        raise RuntimeError(f"ONNX model produced no outputs: {model.model_path}")
    return outputs[0]


def _run_tensorrt(model, input_image, binding_name, input_shape_format):
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT inference requires both tensorrt and pycuda."
        ) from exc

    engine = model.session
    with engine.create_execution_context() as context:
        image_shape = input_image.shape
        if input_shape_format == "CHW":
            if len(image_shape) == 4:
                _, _, image_height, image_width = image_shape
            else:
                _, image_height, image_width = image_shape
            binding_shape = (1, 3, image_height, image_width)
        elif input_shape_format == "HWC":
            if len(image_shape) == 4:
                _, image_height, image_width, _ = image_shape
            else:
                image_height, image_width, _ = image_shape
            binding_shape = (1, image_height, image_width, 3)
        else:
            raise ValueError("Unsupported input_shape_format. Use 'CHW' or 'HWC'.")

        context.set_binding_shape(engine.get_binding_index(binding_name), binding_shape)
        input_buffer = np.ascontiguousarray(input_image.astype("float32"))
        input_memory = cuda.mem_alloc(input_buffer.nbytes)
        output_buffer = None
        output_memory = None
        bindings = []

        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        if output_buffer is None or output_memory is None:
            raise RuntimeError(f"TensorRT model produced no output binding: {model.model_path}")

        stream = cuda.Stream()
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        stream.synchronize()
        return output_buffer


def _segmentation_array(output, image_height, image_width):
    array = np.asarray(output)
    array = np.squeeze(array)

    if array.ndim == 3:
        if array.shape[0] not in (image_height, image_width) and array.shape[0] <= 32:
            array = np.argmax(array, axis=0)
        elif array.shape[-1] <= 32:
            array = np.argmax(array, axis=-1)
        else:
            array = np.squeeze(array)

    if array.size != image_height * image_width:
        array = np.reshape(array, (image_height, image_width))
    else:
        array = array.reshape((image_height, image_width))
    return array


def infer_seg(model, input_file):
    logging.debug(f"Reading input image from file {input_file}")
    with Image.open(input_file) as img:
        orig_size = img.size
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        input_image = preprocess(img)
        image_width = img.width
        image_height = img.height

    if model.backend == "onnx":
        output = _run_onnx(model, np.expand_dims(input_image, axis=0))
    elif model.backend == "tensorrt":
        output = _run_tensorrt(model, input_image, "input", "CHW")
    else:
        raise ValueError(f"Unsupported backend: {model.backend}")

    img = postprocess(_segmentation_array(output, image_height, image_width))
    return img, orig_size


def infer_image(
    model,
    input_file,
    resize_shape,
    binding_name="input",
    input_shape_format="CHW",
    preprocess_func=None,
):
    with Image.open(input_file) as img:
        img = img.resize(resize_shape, Image.Resampling.LANCZOS)
        if preprocess_func:
            input_image = preprocess_func(img)
            onnx_input = np.expand_dims(input_image, axis=0)
        else:
            input_image = np.asarray(img.convert("RGB")).astype("float32")
            onnx_input = np.expand_dims(input_image, axis=0)

    if model.backend == "onnx":
        return _run_onnx(model, onnx_input)
    if model.backend == "tensorrt":
        trt_input = onnx_input if input_shape_format == "HWC" else input_image
        return _run_tensorrt(model, trt_input, binding_name, input_shape_format)
    raise ValueError(f"Unsupported backend: {model.backend}")


def infer_cla(model, input_file):
    return infer_image(
        model,
        input_file,
        (299, 299),
        binding_name="input_1",
        input_shape_format="HWC",
    )


def infer_cla2(model, input_file):
    return infer_image(
        model,
        input_file,
        (224, 224),
        binding_name="input",
        input_shape_format="CHW",
        preprocess_func=preprocess,
    )
