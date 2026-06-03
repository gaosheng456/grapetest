const $ = (id) => document.getElementById(id);

function setMsg(text) {
  $("msg").textContent = text || "";
}

function apiBase() {
  return $("apiBase").value.trim().replace(/\/$/, "");
}

function computeDefaultApiBase() {
  const host = window.location.hostname;
  const proto = window.location.protocol === "https:" ? "https" : "http";
  if (!host || host === "127.0.0.1" || host === "localhost") {
    return "http://127.0.0.1:8000";
  }
  return `${proto}://${host}:8000`;
}

async function postJson(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    let detail = "";
    try {
      const j = await resp.json();
      detail = j?.detail ? String(j.detail) : "";
    } catch {
      detail = await resp.text();
    }
    throw new Error(detail || `HTTP ${resp.status}`);
  }
  return await resp.json();
}

function redirectToApp() {
  window.location.href = "./index.html";
}

function hasToken() {
  return !!localStorage.getItem("auth_token");
}

async function doLogin() {
  const identifier = $("loginId").value.trim();
  const password = $("loginPwd").value;
  if (!identifier || !password) {
    setMsg("请输入账号和密码");
    return;
  }

  setMsg("登录中...");
  const data = await postJson(`${apiBase()}/api/auth/login`, { identifier, password });
  localStorage.setItem("auth_token", data.token);
  localStorage.setItem("auth_identifier", data.identifier || identifier);
  localStorage.setItem("auth_api_base", apiBase());
  setMsg("登录成功，正在跳转...");
  redirectToApp();
}

async function doRegister() {
  const identifier = $("regId").value.trim();
  const password = $("regPwd").value;
  if (!identifier || !password) {
    setMsg("请输入注册账号与密码");
    return;
  }

  setMsg("注册中...");
  await postJson(`${apiBase()}/api/auth/register`, { identifier, password });
  setMsg("注册成功，请使用该账号登录");
  $("loginId").value = identifier;
  $("loginPwd").value = "";
}

// 已登录则直接回到主页面
if (hasToken()) {
  redirectToApp();
}

// 恢复上次后端地址
const prevApi = localStorage.getItem("auth_api_base");
if (prevApi) {
  $("apiBase").value = prevApi;
} else {
  const current = $("apiBase").value.trim();
  if (!current || current === "http://127.0.0.1:8000") {
    $("apiBase").value = computeDefaultApiBase();
  }
}

$("loginId").value = localStorage.getItem("auth_identifier") || "grape";

$("loginBtn").addEventListener("click", () => {
  doLogin().catch((e) => setMsg(`登录失败：${e.message || e}`));
});

$("registerBtn").addEventListener("click", () => {
  doRegister().catch((e) => setMsg(`注册失败：${e.message || e}`));
});
