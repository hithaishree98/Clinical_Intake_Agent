let threadId = null;
let clientMsgId = 0;
let clinicianToken = null;

function $(id) { return document.getElementById(id); }

function showToast(msg, type = "info", duration = 3000) {
  const t = $("toast");
  t.textContent = msg;
  t.className = `toast ${type}`;
  t.classList.remove("hidden");
  setTimeout(() => t.classList.add("hidden"), duration);
}

function setTyping(on) {
  $("typing").classList.toggle("hidden", !on);
  $("sendBtn").disabled = on;
  $("msg").disabled = on;
}

function setStatus(state, text) {
  const el = $("statusText");
  el.textContent = text;
  el.className = "status-text " + (state || "");
}

function escHtml(s) {
  return (s ?? "").toString()
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function addMsg(role, text, type = "") {
  const chat = $("chat");

  // Remove empty state placeholder on first message
  const empty = chat.querySelector(".chat-empty");
  if (empty) empty.remove();

  const cls = ["msg", role, type].filter(Boolean).join(" ");
  chat.insertAdjacentHTML("beforeend", `
    <div class="${cls}">
      <div class="bubble">${escHtml(text)}</div>
    </div>
  `);
  chat.scrollTop = chat.scrollHeight;
}

function updatePhase(status, phase) {
  const badge = $("sessStatus");
  badge.textContent = status || "—";
  badge.className = "status-badge " + (status || "");

  const phaseVal = phase || "—";
  $("phase").textContent = phaseVal;
  $("sidePhase").textContent = phaseVal;
  $("phaseIndicator").classList.remove("hidden");
}

async function waitForReport(jobId) {
  while (true) {
    await new Promise(r => setTimeout(r, 1500));
    const res = await fetch(`/jobs/${jobId}`);
    const job = await res.json();
    if (job.status === "done") {
      await loadReport();
      return;
    }
    if (job.status === "failed") {
      showToast("Report generation failed: " + (job.error || "unknown error"), "error");
      return;
    }
  }
}

async function loadReport() {
  if (!threadId) return;
  const res = await fetch(`/report/${threadId}`);
  if (!res.ok) {
    $("report").textContent = "Report not ready yet.";
    return;
  }
  const j = await res.json();
  const text = j.latest?.report_text || "(empty)";
  const box = $("report");
  box.textContent = text;
  box.classList.add("has-content");
}

async function start() {
  setStatus("", "Starting session...");
  $("startBtn").disabled = true;

  try {
    const res = await fetch("/start", { method: "POST" });
    const j = await res.json();

    threadId = j.thread_id;
    clientMsgId = 0;

    $("tid").textContent = threadId.slice(0, 16) + "...";
    $("sessionLabel").classList.remove("hidden");
    $("chat").innerHTML = "";
    $("report").textContent = "Complete the intake to generate the clinician note.";
    $("report").classList.remove("has-content");
    $("msg").disabled = false;
    $("sendBtn").disabled = false;

    setStatus("active", "Session active");
    updatePhase("active", j.phase || "identity");
    addMsg("assistant", j.reply);
    $("msg").focus();
  } catch (e) {
    setStatus("error", "Failed to start session");
    showToast("Could not start session. Is the server running?", "error");
  } finally {
    $("startBtn").disabled = false;
  }
}

async function sendMsg() {
  if (!threadId) return showToast("Start a session first.", "info");

  const input = $("msg");
  const msg = input.value.trim();
  if (!msg) return;

  input.value = "";
  addMsg("user", msg);
  setTyping(true);

  clientMsgId++;

  const fd = new FormData();
  fd.append("thread_id", threadId);
  fd.append("message", msg);
  fd.append("client_msg_id", String(clientMsgId));

  try {
    const res = await fetch("/chat", { method: "POST", body: fd });
    const j = await res.json();

    const isEmergency = j.status === "escalated";
    addMsg("assistant", j.reply, isEmergency ? "emergency" : "");
    updatePhase(j.status, j.phase);

    if (isEmergency) {
      setStatus("escalated", "Emergency escalation");
      $("msg").disabled = true;
      $("sendBtn").disabled = true;
    } else if (j.status === "error") {
      setStatus("error", "Error");
    } else {
      setStatus("active", "Session active");
    }

    if (j.job_id) {
      waitForReport(j.job_id);
    }

  } catch (e) {
    showToast("Message failed. Please try again.", "error");
    setStatus("error", "Error");
  } finally {
    setTyping(false);
    if (!$("msg").disabled) $("msg").focus();
  }
}

async function clinicianLogin() {
  const pwd = $("clinicianPwd").value.trim();
  if (!pwd) return showToast("Enter the clinician password.", "info");

  const fd = new FormData();
  fd.append("password", pwd);

  try {
    const res = await fetch("/clinician/token", { method: "POST", body: fd });
    if (!res.ok) return showToast("Incorrect password.", "error");

    const j = await res.json();
    clinicianToken = j.access_token;

    $("clinician-login").classList.add("hidden");
    $("clinician-tools").classList.remove("hidden");
    showToast("Logged in as clinician.", "success");
  } catch (e) {
    showToast("Login failed.", "error");
  }
}

async function loadPending() {
  if (!clinicianToken) return showToast("Log in first.", "info");

  try {
    const res = await fetch("/clinician/pending", {
      headers: { "Authorization": `Bearer ${clinicianToken}` }
    });

    if (!res.ok) return showToast("Authentication failed. Token may have expired.", "error");

    const items = await res.json();
    const list = $("escalations-list");
    list.innerHTML = "";

    if (!Array.isArray(items) || items.length === 0) {
      list.innerHTML = '<p style="font-size:12px;color:var(--muted);padding:4px 0">No pending escalations.</p>';
      list.classList.remove("hidden");
      return;
    }

    items.forEach(item => {
      const div = document.createElement("div");
      div.className = `esc-item ${item.kind || ""}`;
      div.innerHTML = `
        <div class="esc-kind">${escHtml(item.kind || "unknown")}</div>
        <div class="esc-meta">
          ${escHtml((item.thread_id || "").slice(0, 12))}...
          &nbsp;|&nbsp;
          ${escHtml(item.created_at || "")}
        </div>
      `;
      div.addEventListener("click", () => {
        $("escThreadId").value = item.thread_id || "";
        $("escId").value = item.esc_id || "";
        $("resolve-form").classList.remove("hidden");
        $("escNote").focus();
      });
      list.appendChild(div);
    });

    list.classList.remove("hidden");
    showToast(`${items.length} pending escalation${items.length !== 1 ? "s" : ""}.`, "info");
  } catch (e) {
    showToast("Failed to load escalations.", "error");
  }
}

async function resolveEsc() {
  if (!clinicianToken) return showToast("Log in first.", "info");

  const t = $("escThreadId").value.trim();
  const e = $("escId").value.trim();
  const note = $("escNote").value.trim() || "Resolved";

  if (!t || !e) return showToast("Select an escalation from the list above.", "info");

  const fd = new FormData();
  fd.append("thread_id", t);
  fd.append("esc_id", e);
  fd.append("nurse_note", note);

  try {
    const res = await fetch("/clinician/resolve", {
      method: "POST",
      body: fd,
      headers: { "Authorization": `Bearer ${clinicianToken}` }
    });

    const j = await res.json();
    if (j.ok) {
      showToast("Escalation resolved.", "success");
      $("resolve-form").classList.add("hidden");
      $("escThreadId").value = "";
      $("escId").value = "";
      $("escNote").value = "";
      await loadPending();
    } else {
      showToast("Failed to resolve escalation.", "error");
    }
  } catch (e) {
    showToast("Request failed.", "error");
  }
}

async function copyTid() {
  const full = threadId;
  if (!full) return;
  await navigator.clipboard.writeText(full);
  showToast("Session ID copied.", "success");
}

window.addEventListener("DOMContentLoaded", () => {
  $("startBtn").addEventListener("click", start);
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); sendMsg(); });
  $("viewReportBtn").addEventListener("click", loadReport);
  $("pendingBtn").addEventListener("click", loadPending);
  $("resolveBtn").addEventListener("click", resolveEsc);
  $("loginBtn").addEventListener("click", clinicianLogin);
  $("copyTidBtn").addEventListener("click", copyTid);

  $("clinicianPwd").addEventListener("keydown", (e) => {
    if (e.key === "Enter") clinicianLogin();
  });
});
