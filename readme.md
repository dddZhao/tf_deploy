## 本地部署

项目默认使用 ONNX Runtime 推理，可在 Windows 和 macOS 上通用。NVIDIA 部署机仍可切换到 TensorRT 后端获取更高性能。

### 模型对应关系

ONNX Runtime 默认从 `ori_model/` 读取模型。TensorRT 后端从 `model/` 读取 engine：

| ONNX 文件 | Engine 文件 | 代码中的用途 |
| --- | --- | --- |
| `ori_model/end2end_tunnelfaceseg.onnx` | `model/seg_tf.engine` | 掌子面分割 |
| `ori_model/classification_posui.onnx` | `model/cla_posui.engine` | 破碎分类 |
| `ori_model/end2end_waterseg.onnx` | `model/seg_water.engine` | 渗/涌水分割 |
| `ori_model/end2end_YTLX.onnx` | `model/cla_ytlx.engine` | 岩体类型分类 |
| `ori_model/end2end_FHCD.onnx` | `model/cla_fhcd.engine` | 风化程度分类 |
| `ori_model/crackseg_segformer.onnx` | `model/crackseg_segformer.engine` | 裂隙识别 |

### 创建本地环境

```bash
conda env create -f environment.yml
conda activate tf_deploy
```

默认后端是 ONNX Runtime CPU：

```bash
python -c "from utils.runtime import load_model; load_model('model/seg_tf.engine')"
```

这里传入的是旧的 engine 路径，默认后端会自动映射到 `ori_model/end2end_tunnelfaceseg.onnx`，所以业务代码不用改调用方式。

### 运行流程

完整流程包括：掌子面分割、透视变换、切片、筛选、裂隙识别、破碎结构分类、渗/涌水分割、岩体类型分类、风化程度分类。

单张图片完整运行：

```bash
conda activate tf_deploy
python TF_process/run.py data/input/pic_1143.png
```

批量完整运行，处理文件夹下所有支持的图片：

```bash
conda activate tf_deploy
python TF_process/run.py data/input
```

仅执行分割、变换、切片和筛选，不运行裂隙识别和后续分类：

```bash
python TF_process/run.py data/input/pic_1143.png --screen_only
python TF_process/run.py data/input --screen_only
```

`batch_process.py` 当前保留为批量筛选入口，等价于对文件夹执行 `run.py --screen_only`：

```bash
python TF_process/batch_process.py data/input
```

可选参数：

```bash
python TF_process/run.py data/input/pic_1143.png --tile_size 512 512 --log_level INFO
```

输出会按原图分文件夹保存，例如 `data/output/pic_1143/`。裂隙识别会读取筛选后的切片文件夹，并输出：

- `pic_1143_black_transformed_crack_tiles/`：每个 selected tile 的 `__mask.png` 和 `__overlay.png`
- `pic_1143_clear_transformed_crack_overlay.png`：叠加到清晰透视变换图上的整图裂隙结果

### 后端切换

默认：

```bash
TF_DEPLOY_BACKEND=onnx python TF_process/run.py data/input/pic_1143.png
```

NVIDIA 机器使用 TensorRT：

```bash
TF_DEPLOY_BACKEND=tensorrt python TF_process/run.py data/input/pic_1143.png
```

ONNX Runtime 默认使用 `CPUExecutionProvider`。如需指定其他 provider：

```bash
TF_DEPLOY_ONNX_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider python TF_process/run.py data/input/pic_1143.png
```

### TensorRT 限制

TensorRT `.engine` 文件需要在最终部署的 NVIDIA 机器上生成，通常不能在 macOS/Apple Silicon 上生成，也不建议在另一台 GPU/CUDA/TensorRT 版本不同的机器上生成后直接拷贝使用。

目标机器需要具备：

- Linux x86_64
- NVIDIA GPU 与可用驱动
- CUDA
- TensorRT，且 `trtexec` 在 `PATH` 中可用

`tensorrt` 与 `pycuda` 不放在默认环境中。请按目标机器上的 CUDA/TensorRT 版本单独安装，并确认：

```bash
trtexec --version
nvidia-smi
python -c "import tensorrt, pycuda; print(tensorrt.__version__)"
```

### 转换 TensorRT engine

在项目根目录执行：

```bash
bash tools/convert_onnx_to_engines.sh
```

脚本默认使用 FP16，并把输出写入 `model/`。如果要用 TensorRT 默认精度（通常为 FP32），可以清空精度参数：

```bash
PRECISION_FLAGS= bash tools/convert_onnx_to_engines.sh
```

如果目标 TensorRT 对静态模型的 shape 参数报错，可以先单独转换某个模型确认输入名与尺寸，例如：

```bash
trtexec --onnx=ori_model/classification_posui.onnx --fp16 --saveEngine=model/cla_posui.engine
```
