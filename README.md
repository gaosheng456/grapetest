# 葡萄语义分割在线检测系统（ONNX + Python + HTML）

本工程是“前后端分离”的最小可用实现：

- 后端：Python + FastAPI，加载 `yolo_seg.onnx`，提供 HTTP 推理接口
- 前端：纯 HTML/JS，通过 `fetch` 调用后端接口，展示 Mask 与叠加图

> 兼容策略：后端会根据 ONNX 输出形状做启发式判断：
> - 若模型输出是 `(1,1,H,W)` / `(1,H,W)` 等，按**语义分割**处理
> - 否则尝试按常见 **YOLOv8-seg (pred + proto)** 的 ONNX 导出格式做实例分割后处理，并合并为一张 mask 展示

---

## 目录结构

- `yolo_seg.onnx`：你的分割模型（放在仓库根目录）
- `backend/`：后端服务
- `frontend/`：前端静态页面
- `0.png`、`1.jpg`：示例图片

---

## 1) 环境准备（Windows）

建议 Python 3.8+。

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

> 你当前工作区若已存在 `.venv`，可跳过创建步骤。

---

## 2) 启动后端（FastAPI）

方式 A：使用脚本

```powershell
.\backend\run_backend.ps1
```

方式 B：手动启动

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

健康检查：

- `GET http://127.0.0.1:8000/api/health`
- `GET http://127.0.0.1:8000/api/model`（查看 ONNX 输入输出形状，便于排查不兼容）

如你的模型不在根目录，使用环境变量指定：

```powershell
$env:GRAPE_MODEL_PATH = "E:\\learn\\grapetest\\yolo_seg.onnx"
.\backend\run_backend.ps1
```

---

## 3) 启动前端（纯 HTML 静态站）

方式 A：使用脚本

```powershell
.\frontend\run_frontend.ps1
```

然后用浏览器打开：

- `http://127.0.0.1:5173/`

> 说明：前端用 Python 内置 `http.server` 启动，避免直接 `file://` 打开时的安全限制。

---

## 4) 接口说明

### `POST /api/predict`

- 请求：`multipart/form-data` 上传图片字段 `file`
- 参数（query string，可选）：
  - `conf`：置信度阈值（实例分割模式使用）默认 0.25
  - `iou`：NMS 阈值（实例分割模式使用）默认 0.45
  - `threshold`：语义分割阈值默认 0.5

返回 JSON：

- `overlay_png_base64`：叠加图 PNG（base64）
- `mask_png_base64`：mask PNG（base64，白色为前景）
- `detections`：若为实例分割模式，返回 box/score/class_id

---

## 5) 常见问题

1. **浏览器报 CORS**
   - 后端已允许 `*`，请确认前端是通过 `http://127.0.0.1:5173` 打开的，而不是 `file://`。

2. **推理时报“输出不符合 YOLOv8-seg 格式”**
   - 先打开 `GET /api/model` 看输出形状。
   - 如果你的 ONNX 是“纯语义分割输出”，通常会是 `(1,1,H,W)`；若不是这两种常见格式，需要按你的导出方式改动后处理逻辑（文件在 `backend/app/yolo_seg.py`）。

3. **速度慢**
   - 当前使用 CPU 推理；如你需要 GPU，需要安装 `onnxruntime-gpu` 并调整 providers。
