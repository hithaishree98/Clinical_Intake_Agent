let threadId = null;
let clientMsgId = 0;

function $(id){ return document.getElementById(id); }

function setStatus(active, text){
  $("dot").className = "dot" + (active ? " on" : "");
  $("statusText").textContent = text;
}

function escHtml(s){
  return (s ?? "").toString()
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function addLine(role, text){
  const chat = $("chat");
  const safe = escHtml(text);
  chat.insertAdjacentHTML("beforeend", `
    <div class="msg ${role}">
      <div class="bubble">${safe}</div>
    </div>
  `);
  chat.scrollTop = chat.scrollHeight;
}

function resetPanels(){
  $("chat").innerHTML = "";
  $("report").textContent = "Start a session, finish intake, then click “View Report”.";
  $("pending").textContent = "Click “Clinician Pending” to load.";
  $("sessStatus").textContent = "—";
  $("phase").textContent = "—";
  $("cmid").textContent = "0";
  $("escThreadId").value = "";
  $("escId").value = "";
  $("escNote").value = "";
}

async function start(){
  setStatus(false, "Starting…");
  const r = await fetch("/start", { method: "POST" });
  const j = await r.json();

  threadId = j.thread_id;
  clientMsgId = 0;

  $("tid").textContent = threadId;
  resetPanels();

  setStatus(true, "Session active");
  $("sessStatus").textContent = "active";
  $("phase").textContent = j.phase || "identity";
  addLine("assistant", j.reply);

  $("msg").focus();
}

async function sendMsg(){
  if(!threadId) return alert("Start a session first");

  const input = $("msg");
  const msg = input.value.trim();
  if(!msg) return;

  input.value = "";
  addLine("user", msg);

  clientMsgId++;
  $("cmid").textContent = String(clientMsgId);

  const fd = new FormData();
  fd.append("thread_id", threadId);
  fd.append("message", msg);
  fd.append("client_msg_id", String(clientMsgId));

  const r = await fetch("/chat", { method: "POST", body: fd });
  const j = await r.json();

  if(j.status === "error"){
    setStatus(false, "Error");
  } else {
    setStatus(true, j.status === "escalated" ? "Escalated" : "Session active");
  }

  $("sessStatus").textContent = j.status || "—";
  $("phase").textContent = j.phase || "—";

  addLine("assistant", j.reply);
}

async function viewReport(){
  if(!threadId) return alert("Start a session first");
  const r = await fetch(`/report/${threadId}`);
  if(!r.ok){
    $("report").textContent = "Report not ready yet.";
    return;
  }
  const j = await r.json();
  $("report").textContent = j.latest?.report_text || "(empty)";
}

async function pending(){
  const r = await fetch("/clinician/pending", {
    headers: { "X-Clinician-Token": "dev-token" }
  });
  const j = await r.json();
  $("pending").textContent = JSON.stringify(j, null, 2);

  // helper: auto-fill resolve inputs if there is one pending
  if(Array.isArray(j) && j.length > 0){
    $("escThreadId").value = j[0].thread_id || "";
    $("escId").value = j[0].esc_id || "";
  }
}

async function resolveEsc(){
  const t = $("escThreadId").value.trim();
  const e = $("escId").value.trim();
  const note = $("escNote").value.trim() || "Resolved";

  if(!t || !e) return alert("Provide thread_id and esc_id");

  const fd = new FormData();
  fd.append("thread_id", t);
  fd.append("esc_id", e);
  fd.append("nurse_note", note);

  const r = await fetch("/clinician/resolve", {
    method: "POST",
    body: fd,
    headers: { "X-Clinician-Token": "dev-token" }
  });
  const j = await r.json();
  alert(j.ok ? "Resolved" : "Done");
}

async function copyTid(){
  const tid = $("tid").textContent;
  if(!tid || tid === "—") return;
  await navigator.clipboard.writeText(tid);
  alert("thread_id copied");
}

window.addEventListener("DOMContentLoaded", () => {
  setStatus(false, "No session");

  $("startBtn").addEventListener("click", start);
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); sendMsg(); });
  $("viewReportBtn").addEventListener("click", viewReport);
  $("pendingBtn").addEventListener("click", pending);
  $("resolveBtn").addEventListener("click", resolveEsc);
  $("copyTidBtn").addEventListener("click", copyTid);
});

async function waitForJob(jobId) {
  while (true) {
    const res = await fetch(`/jobs/${jobId}`);
    const job = await res.json();
    if (job.status === "done") return;
    if (job.status === "failed") throw new Error(job.error || "Job failed");
    await new Promise(r => setTimeout(r, 1500));
  }
}
