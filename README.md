# 葡萄语义分割在线检测系统（ONNX + Python + HTML）

本工程是“前后端分离”的最小可用实现：

- 后端：Python + FastAPI，加载 `yolo_seg.onnx`，提供 HTTP 推理接口
- 前端：纯 HTML/JS，通过 `fetch` 调用后端接口，展示 Mask 与叠加图
- 摄像头：点击“打开摄像头”后，左侧实时显示摄像头画面，右侧实时显示分割叠加结果
- 登录/注册：提供登录页（手机号/邮箱注册），登录后才允许调用推理接口
- 结果累积：多次检测不覆盖表格结果，序号自动按 1…N 连续排序；点击“清空界面”才会清空
- 结果计数：显示“当前图片/当前帧”检测到的葡萄数量，单位为“串”（例如 `2串`）
- 结果导出：点击“结果保存”导出 CSV（含 UTF-8 BOM，便于 Excel 打开）

> 兼容策略：后端会根据 ONNX 输出形状做启发式判断：
> - 若模型输出是 `(1,1,H,W)` / `(1,H,W)` 等，按**语义分割**处理
> - 否则尝试按常见 **YOLOv8-seg (pred + proto)** 的 ONNX 导出格式做实例分割后处理，并合并为一张 mask 展示

---

## 目录结构

- `yolo_seg.onnx`：分割模型（放在仓库根目录）
- `backend/`：后端服务（FastAPI）
- `frontend/`：前端静态页面（HTML/JS/CSS）
- `grape-example*.jpg`：示例图片（如仓库中存在）
- `run_predict_test.py`：可选验证脚本（调用后端一次推理并输出 overlay/mask PNG）
- `backend/app/password_store.json`：账号密码存储文件（后端启动后自动生成，存储加盐哈希）

---

## 1) 环境准备（Windows）

建议 Python 3.8+。

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

> 如果工作区已存在 `.venv`，可跳过创建步骤。

---

## 2) 启动后端（FastAPI）

方式 A：使用脚本（推荐）

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
$env:GRAPE_MODEL_PATH = "E:\learn\grapetest\yolo_seg.onnx"
.\backend\run_backend.ps1
```

---

## 3) 启动前端（静态站）

方式 A：使用脚本（推荐）

```powershell
.\frontend\run_frontend.ps1
```

然后用浏览器打开：

- `http://127.0.0.1:5173/`
- 或直接打开登录页：`http://127.0.0.1:5173/login.html`

> 说明：前端用 Python 内置 `http.server` 启动，避免直接 `file://` 打开时的安全限制。

---

## 4) 使用方法（可视化）

### 先登录

打开 `http://127.0.0.1:5173/` 后会自动跳转到登录页：

- 登录页：`http://127.0.0.1:5173/login.html`
- 默认账号：`grape`
- 默认密码：`123`
- 注册：账号必须是**手机号**或**邮箱**（例如 `13800138000` / `user@example.com`）

登录成功后会自动回到主页面开始检测。

打开 `http://127.0.0.1:5173/` 后，有两种方式：

### A. 图片检测

1. 点击“打开图片”，选择一张图片
   - 也可以点击“打开文件夹”自动选择文件夹内第一张图片
2. 点击“开始图像分割”
3. 页面会显示：
   - 左侧：原始图像
   - 右侧：分割叠加结果
   - 下方表格：按后端返回 `detections` 逐条追加（会累积追加，序号自动重排为 1…N）

备注：

- “检测结果计数”显示的是**当前图片**检测到的葡萄数量（单位：串），不是累计表格行数。

### B. 摄像头实时检测

1. 点击“打开摄像头”，浏览器弹窗选择“允许”摄像头权限
2. 左侧“原始图像”区域会实时显示摄像头画面
3. 右侧“图像分割结果”会按固定间隔抓帧并调用后端 `/api/predict`，实时更新叠加结果与统计
4. 再次点击按钮（显示为“关闭摄像头”）可停止实时检测

注意：

- 摄像头开启时，为避免冲突，手动“开始图像分割”会提示先关闭摄像头
- 选择图片/打开文件夹时会自动关闭摄像头

---

## 5) 参数说明

页面参数会作为 query string 传给后端 `/api/predict`：

- `conf`：置信度阈值（实例分割模式使用）
- `iou`：NMS 阈值（实例分割模式使用）
- `threshold`：语义阈值

默认值：

- 后端接口默认：`conf=0.25`、`iou=0.45`、`threshold=0.5`
- 前端 UI 默认填入：`conf=0.60`、`iou=0.30`、`threshold=0.60`

### 小面积噪声去除（后端默认已开启）

后端会对 mask 做连通域过滤，去除小碎块误识别区域，默认参数：

- `min_area=100000`
- `min_w=40`
- `min_h=40`

这些参数目前不在前端 UI 暴露，但后端接口支持通过 query string 覆盖（见下方接口说明）。

---

## 6) 接口说明

### 登录/注册

- `POST /api/auth/register`
   - JSON：`{"identifier": "手机号或邮箱", "password": "..."}`
- `POST /api/auth/login`
   - JSON：`{"identifier": "...", "password": "..."}`
   - 返回：`{"token": "..."}`
- `GET /api/auth/me`
   - 请求头：`Authorization: Bearer <token>`
   - 返回：当前登录账号

### `POST /api/predict`

- 请求：`multipart/form-data` 上传图片字段 `file`
- 请求头：需要携带 `Authorization: Bearer <token>`（先调用 `/api/auth/login` 获取）
- 参数（query string，可选）：
  - `conf`：置信度阈值（实例分割模式使用）默认 0.25
  - `iou`：NMS 阈值（实例分割模式使用）默认 0.45
  - `threshold`：语义分割阈值默认 0.5
   - `min_area`：连通域最小面积过滤默认 100000
   - `min_w`：连通域外接框最小宽默认 40
   - `min_h`：连通域外接框最小高默认 40

返回 JSON：

- `overlay_png_base64`：叠加图 PNG（base64）
- `mask_png_base64`：mask PNG（base64，白色为前景）
- `detections`：若为实例分割模式，返回 box/score/class_id

---

## 7) 可视化验证脚本（可选）

如果想不经过前端页面，直接验证后端推理是否正常：

```powershell
.\.venv\Scripts\python.exe .\run_predict_test.py
```

脚本会先用默认账号 `grape/123` 登录拿到 token，再上传图片到后端接口，并在根目录生成：

- `overlay_png_base64.png`
- `mask_png_base64.png`

可选参数：

```powershell
# 指定图片 + 连通域过滤参数
.\.venv\Scripts\python.exe .\run_predict_test.py .\grape-example1.jpg 100000 40 40
```

---

## 8) 常见问题

1. 浏览器报 CORS
   - 后端已允许 `*`，请确认前端是通过 `http://127.0.0.1:5173/` 打开的，而不是 `file://`。

2. 推理时报“输出不符合 YOLOv8-seg 格式”
   - 先打开 `GET /api/model` 看输出形状。
   - 如果你的 ONNX 是“纯语义分割输出”，通常会是 `(1,1,H,W)`；若不是这两种常见格式，需要按你的导出方式改动后处理逻辑（在 `backend/app/yolo_seg.py`）。

3. 速度慢
   - 当前使用 CPU 推理；如你需要 GPU，可安装 `onnxruntime-gpu` 并调整 providers。

4. 摄像头打不开/黑屏
   - 建议使用 Chrome/Edge，并确认已允许该站点的摄像头权限。
   - 本地 `http://127.0.0.1` 通常允许 `getUserMedia`；若换成非 localhost 地址，可能需要 https。

5. 如何“退出登录”
   - 当前实现使用浏览器 `localStorage` 保存 token。
   - 如需退出登录：清除站点数据（或手动清理 `localStorage` 里的 `auth_token`），然后刷新页面即可回到登录页。
