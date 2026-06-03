const $ = (id) => document.getElementById(id);

function computeDefaultApiBase() {
  const host = window.location.hostname;
  const proto = window.location.protocol === "https:" ? "https" : "http";
  if (!host || host === "127.0.0.1" || host === "localhost") {
    return "http://127.0.0.1:8000";
  }
  return `${proto}://${host}:8000`;
}

function getAuthToken() {
  return localStorage.getItem("auth_token") || "";
}

function getAuthApiBase() {
  return localStorage.getItem("auth_api_base") || "";
}

function redirectToLogin() {
  window.location.href = "./login.html";
}

function ensureLoggedIn() {
  if (!getAuthToken()) {
    redirectToLogin();
    return false;
  }
  return true;
}

// 主页面必须先登录
if (!ensureLoggedIn()) {
  // 阻止后续脚本继续执行引发报错
  throw new Error("NOT_LOGGED_IN");
}

const state = {
  previewUrl: "",
  overlayUrl: "",
  maskUrl: "",
  resultRows: [],
  cameraStream: null,
  cameraTimer: null,
  cameraBusy: false,
};

// 恢复登录时使用的后端地址
const savedApiBase = getAuthApiBase();
if (savedApiBase && $("apiBase")) {
  $("apiBase").value = savedApiBase;
} else if ($("apiBase")) {
  const current = $("apiBase").value.trim();
  if (!current || current === "http://127.0.0.1:8000") {
    $("apiBase").value = computeDefaultApiBase();
  }
}

const CAMERA_INTERVAL_MS = 800;

function pickBestEffortClose() {
  window.close();
  setTimeout(() => {
    if (!window.closed) {
      window.location.href = "about:blank";
    }
  }, 50);
}

function minimizeUI() {
  $("mainLayout").classList.add("hidden");
  $("floatingExitBtn").classList.add("hidden");
  $("minimizedBar").classList.remove("hidden");
}

function restoreUI() {
  $("mainLayout").classList.remove("hidden");
  $("floatingExitBtn").classList.remove("hidden");
  $("minimizedBar").classList.add("hidden");
}

function imgFromBase64Png(b64) {
  return `data:image/png;base64,${b64}`;
}

function setStatus(text) {
  const value = text || "待检测";
  $("status").textContent = value;
  $("detailStatus").textContent = value;
}

function setEmpty(id, visible) {
  $(id).style.display = visible ? "flex" : "none";
}

function resetImages() {
  $("preview").removeAttribute("src");
  $("overlay").removeAttribute("src");
  $("mask").removeAttribute("src");
  setEmpty("previewEmpty", true);
  setEmpty("overlayEmpty", true);

  const video = $("camera");
  if (video) {
    video.classList.add("hidden");
    try {
      video.pause();
    } catch {
      // ignore
    }
    video.srcObject = null;
  }
}

function isCameraRunning() {
  return !!state.cameraStream;
}

function showCameraPreview() {
  const video = $("camera");
  if (!video) return;
  video.classList.remove("hidden");
  $("preview").classList.add("hidden");
  setEmpty("previewEmpty", false);
}

function showImagePreview() {
  const video = $("camera");
  if (video) video.classList.add("hidden");
  $("preview").classList.remove("hidden");
}

function stopCamera() {
  if (state.cameraTimer) {
    clearInterval(state.cameraTimer);
    state.cameraTimer = null;
  }
  state.cameraBusy = false;

  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((t) => t.stop());
    state.cameraStream = null;
  }

  const video = $("camera");
  if (video) {
    video.srcObject = null;
    video.classList.add("hidden");
  }

  $("openCameraBtn").textContent = "打开摄像头";
  showImagePreview();
}

function dataUrlToBlob(dataUrl) {
  const [meta, b64] = String(dataUrl).split(",");
  const mime = /data:(.*?);base64/.exec(meta)?.[1] || "application/octet-stream";
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Blob([bytes], { type: mime });
}

function videoFrameToPngBlob(video) {
  const canvas = document.createElement("canvas");
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, w, h);
  const dataUrl = canvas.toDataURL("image/png");
  return dataUrlToBlob(dataUrl);
}

async function predictBlob(blob, filename) {
  const apiBase = $("apiBase").value.trim().replace(/\/$/, "");
  const conf = $("conf").value;
  const iou = $("iou").value;
  const threshold = $("threshold").value;
  const url = `${apiBase}/api/predict?conf=${encodeURIComponent(conf)}&iou=${encodeURIComponent(iou)}&threshold=${encodeURIComponent(threshold)}`;

  const form = new FormData();
  form.append("file", blob, filename);

  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};

  const start = performance.now();
  const resp = await fetch(url, { method: "POST", body: form, headers });
  if (!resp.ok) {
    if (resp.status === 401) {
      localStorage.removeItem("auth_token");
      redirectToLogin();
      throw new Error("未登录");
    }
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  const data = await resp.json();
  const elapsed = ((performance.now() - start) / 1000).toFixed(3);
  return { data, elapsed };
}

async function startCameraRealtime() {
  if (isCameraRunning()) return;

  const video = $("camera");
  if (!video) {
    setStatus("页面缺少摄像头预览组件");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    state.cameraStream = stream;
    video.srcObject = stream;
    showCameraPreview();
    $("openCameraBtn").textContent = "关闭摄像头";
    setStatus("摄像头已开启，实时检测中...");

    state.cameraTimer = setInterval(async () => {
      if (!state.cameraStream || state.cameraBusy) return;
      if (video.readyState < 2) return; // HAVE_CURRENT_DATA

      state.cameraBusy = true;
      try {
        const blob = videoFrameToPngBlob(video);
        const { data, elapsed } = await predictBlob(blob, "camera.png");

        state.overlayUrl = imgFromBase64Png(data.overlay_png_base64);
        state.maskUrl = imgFromBase64Png(data.mask_png_base64);
        $("overlay").src = state.overlayUrl;
        $("mask").src = state.maskUrl;
        setEmpty("overlayEmpty", false);

        // 摄像头模式：仅做实时展示与统计，不累积到历史表格。
        const maskImg = new Image();
        maskImg.onload = () => {
          const metrics = computeConnectedComponents(maskImg);
          const coverage = metrics.totalPixels > 0 ? (metrics.foregroundPixels / metrics.totalPixels) * 100 : 0;

          // 摄像头模式：与图片模式一致，累积到历史表格并重新编号。
          appendResultRows(data.detections || []);
          renderAccumulatedTable();

          $("statCount").textContent = `${getCurrentGrapeCount(data.detections, metrics.components)}串`;
          $("statMode").textContent = `${data.mode || "semantic"} (camera)`;
          $("statPixels").textContent = String(metrics.foregroundPixels);
          $("statCoverage").textContent = `${coverage.toFixed(2)}%`;
          $("statLatency").textContent = `${elapsed}s`;

          $("json").textContent = JSON.stringify({
            source: "camera",
            mode: data.mode,
            detections: data.detections,
            current_grape_count: getCurrentGrapeCount(data.detections, metrics.components),
            connected_components: metrics.components,
            foreground_pixels: metrics.foregroundPixels,
            coverage_percent: Number(coverage.toFixed(4)),
            latency_seconds: Number(elapsed),
            accumulated_rows: state.resultRows.length,
          }, null, 2);
        };
        maskImg.src = state.maskUrl;
      } catch (err) {
        console.error(err);
        setStatus(`摄像头检测失败：${err.message || err}`);
      } finally {
        state.cameraBusy = false;
      }
    }, CAMERA_INTERVAL_MS);
  } catch (err) {
    console.error(err);
    setStatus(`无法打开摄像头：${err.message || err}`);
    stopCamera();
  }
}

function resetTable() {
  $("resultTableBody").innerHTML = '<tr class="table-empty"><td colspan="5">暂无检测结果，请先上传图片并执行分割</td></tr>';
}

function downloadDataUrl(dataUrl, filename) {
  if (!dataUrl) return;
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

function downloadText(text, filename, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

function csvEscape(value) {
  const s = String(value ?? "");
  if (/[\r\n\",]/.test(s)) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function exportResultsToCsv() {
  if (!Array.isArray(state.resultRows) || state.resultRows.length === 0) {
    setStatus("暂无可导出的检测结果");
    return;
  }

  const header = ["序号", "预测类别", "像素面积(px^2)", "边界框(xmin,ymin,xmax,ymax)", "置信度"].join(",");
  const lines = [header];

  state.resultRows.forEach((row, idx) => {
    const box = Array.isArray(row.box) ? `[${row.box.join(", ")}]` : "";
    const score = typeof row.score === "number" ? row.score : "";
    const cols = [
      idx + 1,
      "Grape（葡萄）",
      row.area ?? "",
      box,
      score,
    ].map(csvEscape);
    lines.push(cols.join(","));
  });

  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const ts = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;

  // 加 BOM，便于 Excel 正常识别 UTF-8 中文
  const csvText = "\ufeff" + lines.join("\r\n");
  downloadText(csvText, `grape_results_${ts}.csv`, "text/csv;charset=utf-8");
  setStatus("已导出 CSV");
}

function computeConnectedComponents(maskImage) {
  const canvas = document.createElement("canvas");
  canvas.width = maskImage.width;
  canvas.height = maskImage.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(maskImage, 0, 0);

  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const visited = new Uint8Array(width * height);
  const components = [];
  let foregroundPixels = 0;

  const isForeground = (x, y) => {
    const idx = (y * width + x) * 4;
    return data[idx] > 0 || data[idx + 1] > 0 || data[idx + 2] > 0 || data[idx + 3] > 0;
  };

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = y * width + x;
      if (visited[offset] || !isForeground(x, y)) continue;

      let area = 0;
      let minX = x;
      let minY = y;
      let maxX = x;
      let maxY = y;
      const queue = [[x, y]];
      visited[offset] = 1;

      while (queue.length > 0) {
        const [cx, cy] = queue.shift();
        area += 1;
        foregroundPixels += 1;
        if (cx < minX) minX = cx;
        if (cy < minY) minY = cy;
        if (cx > maxX) maxX = cx;
        if (cy > maxY) maxY = cy;

        const neighbors = [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1],
        ];

        for (const [nx, ny] of neighbors) {
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const nextOffset = ny * width + nx;
          if (visited[nextOffset] || !isForeground(nx, ny)) continue;
          visited[nextOffset] = 1;
          queue.push([nx, ny]);
        }
      }

      if (area >= 50) {
        components.push({
          area,
          box: [minX, minY, maxX, maxY],
        });
      }
    }
  }

  return {
    components: components.sort((a, b) => b.area - a.area),
    foregroundPixels,
    totalPixels: width * height,
  };
}

function appendResultRows(detections) {
  const dets = Array.isArray(detections) ? detections : [];

  dets.forEach((det) => {
    const box = Array.isArray(det?.box) ? det.box : null;
    if (!box || box.length !== 4) return;

    state.resultRows.push({
      area: typeof det?.area === "number" ? det.area : null,
      box,
      score: typeof det?.score === "number" ? det.score : null,
    });
  });
}

function getCurrentGrapeCount(detections, components) {
  const dets = Array.isArray(detections) ? detections : [];
  if (dets.length > 0) return dets.length;
  const comps = Array.isArray(components) ? components : [];
  return comps.length;
}

function renderAccumulatedTable() {
  const rows = [];

  state.resultRows.forEach((item, idx) => {
    rows.push(`
      <tr>
        <td>${idx + 1}</td>
        <td>Grape（葡萄）</td>
        <td>${item.area}</td>
        <td>[${item.box.join(", ")}]</td>
        <td>${typeof item.score === "number" ? `${(item.score * 100).toFixed(1)}%` : "-"}</td>
      </tr>
    `);
  });

  if (rows.length === 0) {
    resetTable();
    return;
  }

  $("resultTableBody").innerHTML = rows.join("");
}

function resetDashboard() {
  stopCamera();
  state.previewUrl = "";
  state.overlayUrl = "";
  state.maskUrl = "";
  state.resultRows = [];
  $("file").value = "";
  $("fileNameHint").textContent = "未选择文件";
  $("detailFilename").textContent = "-";
  $("detailApi").textContent = $("apiBase").value.trim() || "-";
  $("statCount").textContent = "0串";
  $("statMode").textContent = "-";
  $("statPixels").textContent = "0";
  $("statCoverage").textContent = "0.00%";
  $("statLatency").textContent = "-";
  $("json").textContent = "";
  resetImages();
  resetTable();
  setStatus("待检测");
}

function useSelectedFile(file) {
  if (!file) return;

  // 选图时关闭摄像头，避免同时占用预览区。
  stopCamera();

  if (state.previewUrl.startsWith("blob:")) {
    URL.revokeObjectURL(state.previewUrl);
  }

  state.previewUrl = URL.createObjectURL(file);
  $("preview").src = state.previewUrl;
  $("fileNameHint").textContent = file.webkitRelativePath || file.name;
  $("detailFilename").textContent = file.webkitRelativePath || file.name;
  setEmpty("previewEmpty", false);
  setEmpty("overlayEmpty", true);
  $("overlay").removeAttribute("src");
  $("mask").removeAttribute("src");
  $("json").textContent = "";
  // 选新图片时不清空历史检测结果；仅更新当前预览与状态。
  setStatus("已选择图片");
}

$("apiBase").addEventListener("input", () => {
  $("detailApi").textContent = $("apiBase").value.trim() || "-";
});

$("file").addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  useSelectedFile(file);
});

$("openImageBtn").addEventListener("click", () => {
  $("file").click();
});

$("openCameraBtn").addEventListener("click", async () => {
  if (isCameraRunning()) {
    stopCamera();
    setStatus("已关闭摄像头");
    return;
  }
  await startCameraRealtime();
});

$("openFolderBtn").addEventListener("click", () => {
  $("folderInput").click();
});

$("folderInput").addEventListener("change", (e) => {
  const files = Array.from(e.target.files || []);
  const imageFile = files.find((file) => /\.(png|jpg|jpeg|bmp|webp)$/i.test(file.name));
  if (!imageFile) {
    setStatus("文件夹中未找到可用图片");
    return;
  }
  useSelectedFile(imageFile);
});

$("run").addEventListener("click", async () => {
  if (isCameraRunning()) {
    setStatus("摄像头实时检测中，请先关闭摄像头再手动检测");
    return;
  }
  const apiBase = $("apiBase").value.trim().replace(/\/$/, "");
  const file = $("file").files?.[0];
  if (!file) {
    setStatus("请先选择图片");
    return;
  }

  const conf = $("conf").value;
  const iou = $("iou").value;
  const threshold = $("threshold").value;
  const form = new FormData();
  form.append("file", file);
  const url = `${apiBase}/api/predict?conf=${encodeURIComponent(conf)}&iou=${encodeURIComponent(iou)}&threshold=${encodeURIComponent(threshold)}`;

  const token = getAuthToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};

  $("run").disabled = true;
  setStatus("检测中...");
  $("detailApi").textContent = apiBase;

  const start = performance.now();
  try {
    const resp = await fetch(url, { method: "POST", body: form, headers });
    if (!resp.ok) {
      if (resp.status === 401) {
        localStorage.removeItem("auth_token");
        redirectToLogin();
        throw new Error("未登录");
      }
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }

    const data = await resp.json();
    const elapsed = ((performance.now() - start) / 1000).toFixed(3);

    state.overlayUrl = imgFromBase64Png(data.overlay_png_base64);
    state.maskUrl = imgFromBase64Png(data.mask_png_base64);

    $("overlay").src = state.overlayUrl;
    $("mask").src = state.maskUrl;
    setEmpty("overlayEmpty", false);

    const maskImg = new Image();
    maskImg.onload = () => {
      const metrics = computeConnectedComponents(maskImg);
      const coverage = metrics.totalPixels > 0 ? (metrics.foregroundPixels / metrics.totalPixels) * 100 : 0;

      // 累积结果：不覆盖上一轮表格，而是追加并重新按 1..N 编号。
      appendResultRows(data.detections || []);
      renderAccumulatedTable();

      $("statCount").textContent = `${getCurrentGrapeCount(data.detections, metrics.components)}串`;
      $("statMode").textContent = data.mode || "semantic";
      $("statPixels").textContent = String(metrics.foregroundPixels);
      $("statCoverage").textContent = `${coverage.toFixed(2)}%`;
      $("statLatency").textContent = `${elapsed}s`;

      $("json").textContent = JSON.stringify({
        filename: data.filename,
        mode: data.mode,
        detections: data.detections,
        current_grape_count: getCurrentGrapeCount(data.detections, metrics.components),
        connected_components: metrics.components,
        foreground_pixels: metrics.foregroundPixels,
        coverage_percent: Number(coverage.toFixed(4)),
        latency_seconds: Number(elapsed),
      }, null, 2);
    };
    maskImg.src = state.maskUrl;

    setStatus("检测完成");
  } catch (err) {
    console.error(err);
    setStatus(`检测失败：${err.message || err}`);
    $("json").textContent = String(err.stack || err.message || err);
  } finally {
    $("run").disabled = false;
  }
});

$("reset").addEventListener("click", resetDashboard);

$("exportCsv").addEventListener("click", exportResultsToCsv);
$("minimizeBtn").addEventListener("click", minimizeUI);
$("restoreBtn").addEventListener("click", restoreUI);
$("closeBtn").addEventListener("click", pickBestEffortClose);
$("floatingExitBtn").addEventListener("click", pickBestEffortClose);

resetDashboard();
