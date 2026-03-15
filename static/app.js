let threadId    = null;
let clientMsgId = 0;
let clinicianToken = null;

function $(id) { return document.getElementById(id); }

/* ── Toast ─────────────────────────────────────────────── */
function showToast(msg, type = "info", duration = 3000) {
  const t = $("toast");
  t.textContent = msg;
  t.className = `toast ${type}`;
  t.classList.remove("hidden");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), duration);
}

/* ── UI helpers ────────────────────────────────────────── */
function setTyping(on) {
  $("typing").classList.toggle("hidden", !on);
  $("sendBtn").disabled = on;
  $("msg").disabled     = on;
}

function setStatus(state, text) {
  $("statusText").textContent = text;
  $("statusText").className   = "status-text " + (state || "");
  $("statusDot").className    = "status-dot "  + (state || "");
}

function escHtml(s) {
  return (s ?? "").toString()
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/* ── Phase tracker ─────────────────────────────────────── */
const PHASE_ORDER = ["consent","identity","identity_review","subjective","clinical_history","confirm","done"];

function updatePhase(status, phase) {
  // sidebar
  const badge = $("sessStatus");
  badge.textContent = status || "—";
  badge.className   = "badge " + (status || "");

  $("sidePhase").textContent = (phase || "—").replace(/_/g, " ");

  // top phase track
  const steps    = document.querySelectorAll(".phase-step");
  const current  = PHASE_ORDER.indexOf(phase || "");

  steps.forEach(step => {
    const idx = PHASE_ORDER.indexOf(step.dataset.phase);
    step.classList.remove("active-phase", "done-phase");
    if (idx === current)       step.classList.add("active-phase");
    else if (idx < current)    step.classList.add("done-phase");
  });
}

/* ── Messages ──────────────────────────────────────────── */
function addMsg(role, text, type = "") {
  const chat  = $("chat");
  const empty = chat.querySelector(".chat-empty");
  if (empty) empty.remove();

  const cls = ["msg", role, type].filter(Boolean).join(" ");
  chat.insertAdjacentHTML("beforeend",
    `<div class="${cls}"><div class="bubble">${escHtml(text)}</div></div>`
  );
  chat.scrollTop = chat.scrollHeight;
}

/* ── Report polling ────────────────────────────────────── */
async function waitForReport(jobId) {
  while (true) {
    await new Promise(r => setTimeout(r, 1500));
    const res = await fetch(`/jobs/${jobId}`);
    const job = await res.json();
    if (job.status === "done")   { await loadReport(); return; }
    if (job.status === "failed") {
      showToast("Report generation failed: " + (job.error || "unknown"), "error");
      return;
    }
  }
}

async function loadReport() {
  if (!threadId) return;
  const res = await fetch(`/report/${threadId}`);
  if (!res.ok) { $("report").textContent = "Report not ready yet."; return; }
  const j    = await res.json();
  const box  = $("report");
  box.textContent = j.latest?.report_text || "(empty)";
  box.classList.add("has-content");
}

/* ── Start session ─────────────────────────────────────── */
async function start() {
  setStatus("", "Starting…");
  $("startBtn").disabled = true;

  try {
    const res = await fetch("/start", { method: "POST" });
    const j   = await res.json();

    threadId    = j.thread_id;
    clientMsgId = 0;

    $("tid").textContent = threadId.slice(0, 12) + "…";
    $("sessionLabel").classList.remove("hidden");
    $("chat").innerHTML  = "";
    const box = $("report");
    box.textContent = "Complete the intake to generate the clinician note.";
    box.classList.remove("has-content");

    $("msg").disabled     = false;
    $("sendBtn").disabled = false;

    setStatus("active", "Session active");
    updatePhase("active", j.phase || "identity");
    addMsg("assistant", j.reply);
    $("msg").focus();
  } catch {
    setStatus("error", "Failed to start");
    showToast("Could not start session. Is the server running?", "error");
  } finally {
    $("startBtn").disabled = false;
  }
}

/* ── Send message ──────────────────────────────────────── */
async function sendMsg() {
  if (!threadId) return showToast("Start a session first.", "info");
  const input = $("msg");
  const msg   = input.value.trim();
  if (!msg) return;

  input.value = "";
  addMsg("user", msg);
  setTyping(true);
  clientMsgId++;

  const fd = new FormData();
  fd.append("thread_id",    threadId);
  fd.append("message",      msg);
  fd.append("client_msg_id", String(clientMsgId));

  try {
    const res = await fetch("/chat", { method: "POST", body: fd });
    const j   = await res.json();
    const isEmergency = j.status === "escalated";

    addMsg("assistant", j.reply, isEmergency ? "emergency" : "");
    updatePhase(j.status, j.phase);

    if (isEmergency) {
      setStatus("escalated", "Emergency escalation");
      $("msg").disabled     = true;
      $("sendBtn").disabled = true;
    } else if (j.status === "error") {
      setStatus("error", "Error");
    } else if (j.phase === "done") {
      setStatus("done", "Intake complete");
    } else {
      setStatus("active", "Session active");
    }

    if (j.job_id) waitForReport(j.job_id);

  } catch {
    showToast("Message failed. Please try again.", "error");
    setStatus("error", "Error");
  } finally {
    setTyping(false);
    if (!$("msg").disabled) $("msg").focus();
  }
}

/* ── Clinician auth ────────────────────────────────────── */
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
    showToast("Authenticated as clinician.", "success");
  } catch {
    showToast("Login failed.", "error");
  }
}

/* ── Pending escalations ───────────────────────────────── */
async function loadPending() {
  if (!clinicianToken) return showToast("Log in first.", "info");

  try {
    const res = await fetch("/clinician/pending", {
      headers: { "Authorization": `Bearer ${clinicianToken}` }
    });
    if (!res.ok) return showToast("Auth failed. Token may have expired.", "error");

    const items = await res.json();
    const list  = $("escalations-list");
    list.innerHTML = "";

    if (!Array.isArray(items) || items.length === 0) {
      list.innerHTML = '<p style="font-family:var(--mono);font-size:11px;color:var(--text-dim);padding:8px 0">No pending escalations.</p>';
      list.classList.remove("hidden");
      return;
    }

    items.forEach(item => {
      const div = document.createElement("div");
      div.className = `esc-item ${item.kind || ""}`;
      div.innerHTML = `
        <div class="esc-kind">${escHtml(item.kind || "unknown")}</div>
        <div class="esc-meta">${escHtml((item.thread_id || "").slice(0, 12))}… &middot; ${escHtml(item.created_at || "")}</div>
      `;
      div.addEventListener("click", () => {
        $("escThreadId").value = item.thread_id || "";
        $("escId").value       = item.esc_id    || "";
        $("resolve-form").classList.remove("hidden");
        $("escNote").focus();
      });
      list.appendChild(div);
    });

    list.classList.remove("hidden");
    showToast(`${items.length} escalation${items.length !== 1 ? "s" : ""} pending.`, "info");
  } catch {
    showToast("Failed to load escalations.", "error");
  }
}

/* ── Resolve escalation ────────────────────────────────── */
async function resolveEsc() {
  if (!clinicianToken) return showToast("Log in first.", "info");
  const t    = $("escThreadId").value.trim();
  const e    = $("escId").value.trim();
  const note = $("escNote").value.trim() || "Resolved";
  if (!t || !e) return showToast("Select an escalation from the list.", "info");

  const fd = new FormData();
  fd.append("thread_id",  t);
  fd.append("esc_id",     e);
  fd.append("nurse_note", note);

  try {
    const res = await fetch("/clinician/resolve", {
      method: "POST", body: fd,
      headers: { "Authorization": `Bearer ${clinicianToken}` }
    });
    const j = await res.json();
    if (j.ok) {
      showToast("Escalation resolved.", "success");
      $("resolve-form").classList.add("hidden");
      [$("escThreadId"), $("escId"), $("escNote")].forEach(el => el.value = "");
      await loadPending();
    } else {
      showToast("Failed to resolve.", "error");
    }
  } catch {
    showToast("Request failed.", "error");
  }
}

/* ── Copy session ID ───────────────────────────────────── */
async function copyTid() {
  if (!threadId) return;
  await navigator.clipboard.writeText(threadId);
  showToast("Session ID copied.", "success");
}

/* ── Event bindings ────────────────────────────────────── */
window.addEventListener("DOMContentLoaded", () => {
  $("startBtn").addEventListener("click", start);
  $("sendBtn").addEventListener("click", sendMsg);
  $("msg").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMsg(); } });
  $("viewReportBtn").addEventListener("click", loadReport);
  $("pendingBtn").addEventListener("click", loadPending);
  $("resolveBtn").addEventListener("click", resolveEsc);
  $("loginBtn").addEventListener("click", clinicianLogin);
  $("copyTidBtn").addEventListener("click", copyTid);
  $("clinicianPwd").addEventListener("keydown", e => { if (e.key === "Enter") clinicianLogin(); });
});