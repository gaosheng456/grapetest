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
- `start_release.bat`：发布版一键启动（自动建 venv/装依赖/启动前后端）
- `stop_release.bat`：发布版一键停止（按端口结束 8000/5173）
- `backend/app/password_store.json`：账号密码存储文件（后端启动后自动生成，存储加盐哈希；建议不要提交）
- `backend/app/auth_secret.txt`：Token 签名密钥（首次登录/签发 token 时自动生成；建议不要提交）

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
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

健康检查：

- `GET http://127.0.0.1:8000/api/health`
- `GET http://127.0.0.1:8000/api/model`（查看 ONNX 输入输出形状，便于排查不兼容）

说明：

- 后端默认监听 `0.0.0.0:8000`，因此同一局域网可通过 `http://<你的IP>:8000` 访问。

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

- 本机访问：`http://127.0.0.1:5173/`
- 或直接打开登录页：`http://127.0.0.1:5173/login.html`

说明：

- 前端默认监听 `0.0.0.0:5173`，因此同一局域网可通过 `http://<你的IP>:5173/` 访问。

> 说明：前端用 Python 内置 `http.server` 启动，避免直接 `file://` 打开时的安全限制。

---

## 4) 使用方法（可视化）

### 先登录

打开 `http://127.0.0.1:5173/` 后会自动跳转到登录页（未登录无法进入主页面）：

- 登录页：`http://127.0.0.1:5173/login.html`
- 默认账号：`grape`
- 默认密码：`123`
- 注册：账号必须是**手机号**或**邮箱**（例如 `13800138000` / `user@example.com`）

登录成功后会自动回到主页面开始检测。

补充：

- 登录页里的 `API Base` 会尽量自动按当前访问的主机名填入（例如通过 `http://192.168.1.23:5173/` 打开时，会默认填 `http://192.168.1.23:8000`）。

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

---

## 9) 把网址发给别人（局域网访问）

如果你想把网页链接发给同一局域网（同 Wi-Fi/同路由器）的其他人使用，需要确保：

- 运行后端时监听 `0.0.0.0:8000`（脚本已默认配置）
- 运行前端静态站监听 `0.0.0.0:5173`（脚本已默认配置）
- Windows 防火墙允许入站访问 5173/8000 端口

步骤：

1) 在服务器这台电脑上启动后端与前端：

```powershell
.\backend\run_backend.ps1
.\frontend\run_frontend.ps1
```

2) 查出这台电脑的局域网 IP（IPv4）：

```powershell
ipconfig
```

例如查到是 `192.168.1.23`，那么把下面网址发给别人：

- `http://192.168.1.23:5173/`

3) 对方打开后会进入登录页；登录页里的 `API Base` 会默认自动填成 `http://192.168.1.23:8000`。
   - 如果没有自动填，手动改成：`http://192.168.1.23:8000`

4) 如果对方打不开页面/登录失败，通常是防火墙拦截。可用管理员 PowerShell 放行端口：

```powershell
New-NetFirewallRule -DisplayName "grapetest-frontend-5173" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 5173
New-NetFirewallRule -DisplayName "grapetest-backend-8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000
```

注意：

- “打开摄像头”在非 `localhost/127.0.0.1` 的 **http** 页面上，很多浏览器会因为“非安全上下文”而禁止（需要 https）。
  - 图片上传检测不受影响；如果确实要远程用摄像头，建议用 https 反向代理或内网穿透（会超出本工程最小实现范围）。

---

## 10) 发布版（发给别人双击可用）

如果你希望把整个工程文件夹打包发给别人，在 Windows 上“解压后双击就能跑”，可以使用仓库根目录的发布脚本：

- 启动：`start_release.bat`
- 停止：`stop_release.bat`

发布版特点：

- 自动创建 `.venv`（若不存在）并安装依赖
- 后端以“非 reload”模式启动（更接近发布运行方式）
- 前端静态站对外监听，便于局域网内访问

使用步骤：

1) 把整个文件夹发给对方（建议 zip 压缩后发送），对方解压到任意目录。
2) 对方双击运行 `start_release.bat`。
3) 浏览器打开 `http://127.0.0.1:5173/`（或局域网访问 `http://<IP>:5173/`）。
4) 默认登录：`grape / 123`。

说明：

- 发布版首次运行会安装依赖，时间较长属正常。
- 后端鉴权密钥会在首次登录/签发 token 时生成并保存在 `backend/app/auth_secret.txt`（可用环境变量 `GRAPE_AUTH_SECRET` 覆盖）。

---

## 11) 工程实现过程（从 0 到可发布）

本节用于记录本工程“逐步实现/迭代”的关键节点，方便你或其他人二次开发时快速理解整体架构与每一步是怎么做出来的。

### 11.1 初版：后端推理 API + 前端可视化（最小可用）

目标：实现“上传图片 → 后端 ONNX 推理 → 前端展示 mask 与叠加图”。

后端（FastAPI）：

- 入口：`backend/app/main.py`
   - `startup` 时加载 ONNX（默认读取仓库根目录 `yolo_seg.onnx`，也支持环境变量 `GRAPE_MODEL_PATH` 指定）
   - 提供基础接口：
      - `GET /api/health`：健康检查
      - `GET /api/model`：输出 ONNX 输入输出信息，便于排查模型不兼容
      - `POST /api/predict`：接收图片并返回推理结果
- 推理：`backend/app/inference.py`
   - 封装 `OnnxGrapeSegmenter`，负责：预处理 → ONNX 推理 → 后处理 → 生成 mask/overlay
- 工具：`backend/app/utils.py`
   - `imdecode_image(...)`：把上传文件解码成 OpenCV BGR
   - `encode_png_base64(...)`：把图片编码成 base64 PNG 给前端展示

前端（静态 HTML/JS）：

- 页面：`frontend/index.html`
   - 控制面板（后端地址、阈值等）
   - 可视化区（原图、叠加图）
   - 结果表格/JSON 预览
- 逻辑：`frontend/app.js`
   - `fetch` 调用 `/api/predict`，把返回的 `overlay_png_base64` / `mask_png_base64` 直接渲染到 `<img>`

### 11.2 前端迭代：结果累积、计数修正、CSV 导出

目标：让系统更“像一个可用工具”，而不仅是一次性演示。

主要改动点（`frontend/app.js`）：

- 结果累积：多次检测不清空上次表格，统一维护 `state.resultRows` 追加渲染
   - 序号自动重排为 1…N
   - 仅“清空界面”按钮才会清空
- 计数口径修正：
   - “检测结果计数”改为显示**当前图片/当前帧**葡萄数量，并加单位“串”（例如 `2串`）
   - 不再用“累计历史行数”冒充当前计数
- 表格数据源一致性：
   - 表格行严格按后端返回的 `detections` 追加（保证：表格行数 = 葡萄数量）
- CSV 导出：
   - “结果保存”按钮导出表格为 CSV
   - 文件写入 UTF-8 BOM，方便 Excel 直接打开不乱码

同时，UI 默认阈值调整为更适合当前场景：

- `conf=0.60`、`iou=0.30`、`threshold=0.60`（前端输入框默认值）

### 11.3 摄像头实时检测（前端）

目标：支持“打开摄像头 → 周期抓帧 → 实时推理”，并与图片检测一样把结果追加到表格。

主要实现（`frontend/app.js`）：

- `navigator.mediaDevices.getUserMedia(...)` 获取视频流
- `setInterval` 定时抓帧（canvas 截图 → PNG Blob）
- 复用同一个 `predictBlob(...)` 走后端 `/api/predict`
- 摄像头开启时避免并发：用 `state.cameraBusy` 做互斥

已知限制（浏览器安全策略）：

- 非 `localhost/127.0.0.1` 的 **http** 页面上，很多浏览器会禁用摄像头（需要 https）

### 11.4 后端迭代：去除小面积误识别（连通域过滤）

目标：减少碎片噪声误识别，提升可用性。

主要实现：

- `backend/app/utils.py`
   - `remove_small_connected_components(mask_u8, min_area, min_w, min_h)`
   - 使用 `connectedComponentsWithStats` 过滤小面积/小宽高连通域
- `backend/app/inference.py`
   - 在生成最终 mask 前应用过滤
   - 把过滤参数暴露为 `/api/predict` 的 query string：`min_area/min_w/min_h`
   - 默认值偏“去噪”：`min_area=100000, min_w=40, min_h=40`

### 11.5 叠加图增强：画框 + 显示置信度

目标：让可视化结果更直观。

主要实现（`backend/app/inference.py`）：

- 在 overlay 上绘制 bbox（绿色框）
- 在框上绘制文本：`Grape 0.87`（置信度）
   - 实例分割：直接用模型 detection score
   - 语义分割：根据概率图在连通域内取均值作为该区域置信度（需要概率图支持）

### 11.6 登录/注册与鉴权：先验证再允许推理

目标：实现“本工程首先需要登录，验证后才能使用”，并保护推理接口。

后端（FastAPI）：

- 账号存储：`backend/app/auth_store.py`
   - 密码使用 PBKDF2（加盐哈希）存入独立文件 `backend/app/password_store.json`
   - 启动时自动初始化默认账号：`grape / 123`
   - 注册仅允许手机号或邮箱格式
- Token：`backend/app/auth_tokens.py`
   - HMAC SHA256 签名的轻量 token
   - secret 支持 `GRAPE_AUTH_SECRET` 环境变量覆盖
   - 若未配置环境变量，会在首次签发 token 时自动生成并写入 `backend/app/auth_secret.txt`
- 路由：`backend/app/auth_routes.py`
   - `POST /api/auth/register`
   - `POST /api/auth/login` → 返回 token
   - `GET /api/auth/me`
- 保护推理接口：`backend/app/main.py`
   - `/api/predict` 增加 `Depends(require_user)`，必须 `Authorization: Bearer <token>`

前端：

- 登录页：`frontend/login.html` + `frontend/login.js`
   - 登录成功后保存 `auth_token`、`auth_api_base`，并跳转回主页面
- 主页面强制登录：
   - `frontend/index.html` 在渲染前检测 token，无 token 直接跳转登录页
   - `frontend/app.js` 调用 `/api/predict` 自动附带 Bearer Token；遇到 401 会清 token 并跳转登录

### 11.7 可分享（局域网）与可发布（一键启动）

目标：把工程发给别人，能够成功打开网页、登录并检测。

局域网分享：

- 后端监听 `0.0.0.0:8000`：`backend/run_backend.ps1`
- 前端监听 `0.0.0.0:5173`：`frontend/run_frontend.ps1`
- 前端默认 API Base 自动推断：
   - `frontend/login.js` / `frontend/app.js` 通过 `location.hostname` 生成默认 `http://<host>:8000`

发布版脚本：

- `start_release.bat`
   - 自动创建 `.venv`、安装依赖、启动后端与前端
- `stop_release.bat`
   - 按端口结束进程（8000/5173）
- `.gitignore`
   - 避免误提交本地敏感/运行时文件（`.venv/`、密码文件、auth secret 等）

---

如果你希望我把这一章再扩展成“每一步对应的提交点/验收标准”（例如每个阶段应当看到哪些 UI/接口返回什么），我也可以继续补全。
