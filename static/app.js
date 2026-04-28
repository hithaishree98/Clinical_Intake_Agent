let threadId       = null;
let sessionToken   = null;   // bearer token issued by /start, required on /chat
let clientMsgId    = 0;
let currentPhase   = null;   // last known phase from /chat or /start; used by VoiceController

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
  // Mic shares the lock so a fast double-click during /chat or /transcribe
  // can't queue a second recording — only matters once voice is enabled,
  // and the optional-chain keeps this safe pre-DOM-ready or in tests.
  const mic = $("micBtn");
  if (mic && !mic.classList.contains("hidden")) mic.disabled = on;
  // Quick-reply buttons share the disabled state of the composer so a fast
  // double-click can't fire two /chat requests on the same turn.  We toggle
  // the entire .quick-replies container's children rather than tracking
  // individual buttons because new ones may be rendered between calls.
  document.querySelectorAll(".btn-quick-reply").forEach(b => { b.disabled = on; });
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

/* ── Quick-reply buttons ─────────────────────────────────
   Rendered when the server returns `quick_replies` on consent,
   identity_review, or confirm turns.  Each button sends a short canonical
   payload that the server's intent fast-path recognises with zero LLM
   cost.  Free-text input remains available alongside.

   Buttons are removed once one is clicked or the patient types something
   so old prompts don't accumulate. */
function clearQuickReplies() {
  const existing = $("chat").querySelectorAll(".quick-replies");
  existing.forEach(el => el.remove());
}

function renderQuickReplies(replies) {
  if (!Array.isArray(replies) || replies.length === 0) return;
  clearQuickReplies();
  const wrap = document.createElement("div");
  wrap.className = "quick-replies";
  replies.forEach(qr => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-quick-reply";
    btn.textContent = qr.label;
    btn.dataset.payload = qr.payload;
    btn.addEventListener("click", () => {
      // Behave exactly as if the patient typed and pressed Enter.
      $("msg").value = qr.payload;
      sendMsg();
    });
    wrap.appendChild(btn);
  });
  $("chat").appendChild(wrap);
  $("chat").scrollTop = $("chat").scrollHeight;
}

/* ── Voice (Groq Whisper STT + browser TTS) ───────────────
   Voice is intentionally a thin shell on top of the existing /chat
   contract.  Audio is captured locally via MediaRecorder, POSTed to
   /transcribe, the returned text is routed through the same sendMsg()
   path the keyboard uses.  No new state machine, no new safety layer —
   every existing /chat guardrail (idempotency, prompt-injection check,
   PHI masking, cost cap) applies automatically.

   Two safety features specific to voice:
     1. CONFIRM_PHASES — on identity / clinical_history turns the
        transcript is shown for review before sending.  Mishearings on
        names, drug names, and allergies are clinically dangerous; the
        existing identity_review / validate gates downstream catch the
        rest, but stopping the bad text upstream is cheaper.
     2. Hallucination filter on the server returns "" for canned
        Whisper phrases ("Thanks for watching!", etc.); we treat that
        as a soft "didn't catch that" and prompt the patient to retry.

   TTS uses the browser's free speechSynthesis — no Groq cost, audio
   never leaves the device.  Toggle persists in localStorage. */

const VOICE_CONFIRM_PHASES = new Set(["identity", "clinical_history"]);
let voiceCfg          = { enabled: false, max_seconds: 30 };
let mediaRecorder     = null;
let recordedChunks    = [];
let recordingTimer    = null;
let recordingStream   = null;
let ttsEnabled        = localStorage.getItem("tts") === "1";

function voiceSupported() {
  return !!(navigator.mediaDevices &&
            navigator.mediaDevices.getUserMedia &&
            window.MediaRecorder);
}

function ttsSupported() {
  return typeof window.speechSynthesis !== "undefined";
}

async function loadVoiceConfig() {
  try {
    const res = await fetch("/voice/config");
    if (!res.ok) return;
    voiceCfg = await res.json();
  } catch { /* leave disabled */ }

  if (voiceCfg.enabled && voiceSupported()) {
    $("micBtn").classList.remove("hidden");
  }
  if (ttsSupported()) {
    const btn = $("ttsToggleBtn");
    btn.classList.remove("hidden");
    btn.setAttribute("aria-pressed", String(ttsEnabled));
  }
}

function speakReply(text) {
  if (!ttsEnabled || !ttsSupported() || !text) return;
  // Cancel any in-flight utterance — overlapping voices on a new turn
  // is the #1 voice-bot bug.
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.0;
  u.pitch = 1.0;
  speechSynthesis.speak(u);
}

function toggleTts() {
  ttsEnabled = !ttsEnabled;
  localStorage.setItem("tts", ttsEnabled ? "1" : "0");
  $("ttsToggleBtn").setAttribute("aria-pressed", String(ttsEnabled));
  if (!ttsEnabled) speechSynthesis.cancel();
  showToast(ttsEnabled ? "Voice replies on" : "Voice replies off", "info", 1500);
}

async function startRecording() {
  if (!threadId)             return showToast("Start a session first.", "info");
  if (!voiceCfg.enabled)     return;
  if (mediaRecorder)         return; // already recording

  try {
    recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    showToast("Microphone access denied — please type instead.", "error");
    $("micBtn").disabled = true;
    return;
  }

  // Browsers pick a supported codec automatically when mimeType is omitted.
  // Chrome → webm/opus, Safari → mp4/aac, Firefox → ogg/opus.  All three
  // are accepted by /transcribe.
  recordedChunks = [];
  try {
    mediaRecorder = new MediaRecorder(recordingStream);
  } catch {
    cleanupRecording();
    showToast("Voice recording not supported in this browser.", "error");
    return;
  }
  mediaRecorder.ondataavailable = e => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = handleRecordingStopped;
  mediaRecorder.start();

  $("micBtn").classList.add("recording");
  // Hard ceiling: stop automatically after max_seconds even if the user
  // forgets to release the button.  Server enforces the same cap; this
  // just avoids a wasted round-trip.
  recordingTimer = setTimeout(() => stopRecording(), voiceCfg.max_seconds * 1000);
}

function stopRecording() {
  if (!mediaRecorder) return;
  try { mediaRecorder.stop(); } catch { /* already stopped */ }
}

function cleanupRecording() {
  $("micBtn").classList.remove("recording");
  if (recordingTimer) { clearTimeout(recordingTimer); recordingTimer = null; }
  if (recordingStream) {
    recordingStream.getTracks().forEach(t => t.stop());
    recordingStream = null;
  }
  mediaRecorder = null;
}

async function handleRecordingStopped() {
  const chunks = recordedChunks;
  recordedChunks = [];
  cleanupRecording();

  if (chunks.length === 0) return;
  const mime = chunks[0].type || "audio/webm";
  const blob = new Blob(chunks, { type: mime });

  // Tiny blobs are almost certainly accidental clicks — skip the network
  // round-trip and tell the patient.  Threshold is generous: a 200ms
  // clip at 32kbps is ~800 bytes, real speech is several KB minimum.
  if (blob.size < 1500) {
    showToast("Recording too short — hold the mic and speak.", "info");
    return;
  }

  setTyping(true);
  try {
    const fd = new FormData();
    fd.append("thread_id", threadId);
    fd.append("audio",     blob, "audio." + (mime.split("/")[1] || "webm").split(";")[0]);
    const res = await fetch("/transcribe", {
      method: "POST", body: fd,
      headers: { "Authorization": `Bearer ${sessionToken}` },
    });
    if (!res.ok) {
      if (res.status === 413)      showToast("Recording too long — try again.", "error");
      else if (res.status === 415) showToast("Audio format not supported.", "error");
      else if (res.status === 503) showToast("Voice unavailable — please type.", "error");
      else                          showToast("Couldn't transcribe — try again.", "error");
      return;
    }
    const j = await res.json();
    const text = (j.text || "").trim();
    if (!text) {
      showToast("Didn't catch that — try again.", "info");
      return;
    }
    routeTranscript(text);
  } catch {
    showToast("Voice request failed — please type.", "error");
  } finally {
    setTyping(false);
  }
}

function routeTranscript(text) {
  // On clinical-risk phases the patient must review what Whisper heard
  // before it commits.  Everywhere else the transcript auto-sends — the
  // existing quick-reply flow + extraction quality retries already
  // absorb the typical voice errors on those turns.
  if (VOICE_CONFIRM_PHASES.has(currentPhase || "")) {
    showVoiceConfirm(text);
  } else {
    $("msg").value = text;
    sendMsg();
  }
}

function showVoiceConfirm(text) {
  $("voiceConfirmText").textContent = text;
  $("voiceConfirm").classList.remove("hidden");
}

function hideVoiceConfirm() {
  $("voiceConfirm").classList.add("hidden");
  $("voiceConfirmText").textContent = "";
}

function confirmVoiceSend() {
  const text = $("voiceConfirmText").textContent.trim();
  hideVoiceConfirm();
  if (!text) return;
  $("msg").value = text;
  sendMsg();
}

function editVoiceTranscript() {
  const text = $("voiceConfirmText").textContent.trim();
  hideVoiceConfirm();
  $("msg").value = text;
  $("msg").focus();
}

async function loadReport() {
  // Report is available to clinicians via /clinician/case/{thread_id}.
  // The patient sees the note inline in the final chat message from report_node.
}

/* ── Start session ─────────────────────────────────────── */
async function start() {
  localStorage.removeItem("threadId");
  localStorage.removeItem("sessionToken");
  setStatus("", "Starting…");
  $("startBtn").disabled = true;

  try {
    const res = await fetch("/start", { method: "POST" });
    const j   = await res.json();

    threadId     = j.thread_id;
    sessionToken = j.session_token;
    localStorage.setItem("threadId",     threadId);
    localStorage.setItem("sessionToken", sessionToken);
    clientMsgId  = 0;

    $("tid").textContent = threadId.slice(0, 12) + "…";
    $("sessionLabel").classList.remove("hidden");
    $("chat").innerHTML  = "";

    $("msg").disabled     = false;
    $("sendBtn").disabled = false;

    currentPhase = j.phase || "identity";
    setStatus("active", "Session active");
    updatePhase("active", currentPhase);
    addMsg("assistant", j.reply);
    speakReply(j.reply);
    renderQuickReplies(j.quick_replies);
    if ($("micBtn") && voiceCfg.enabled && voiceSupported()) {
      $("micBtn").disabled = false;
    }
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
  // Drop any stale quick-reply buttons or voice-confirm strip from the
  // previous turn so the patient doesn't see stale options after they've
  // moved on.
  clearQuickReplies();
  hideVoiceConfirm();
  addMsg("user", msg);
  setTyping(true);
  clientMsgId++;

  const fd = new FormData();
  fd.append("thread_id",    threadId);
  fd.append("message",      msg);
  fd.append("client_msg_id", String(clientMsgId));

  try {
    const res = await fetch("/chat", {
      method: "POST", body: fd,
      headers: { "Authorization": `Bearer ${sessionToken}` },
    });
    const j   = await res.json();
    const isEmergency = j.status === "escalated";

    addMsg("assistant", j.reply, isEmergency ? "emergency" : "");
    speakReply(j.reply);
    renderQuickReplies(j.quick_replies);
    currentPhase = j.phase;
    if (j.status === "done" || j.status === "escalated") {
      localStorage.removeItem("threadId");
      localStorage.removeItem("sessionToken");
    }
    updatePhase(j.status, j.phase);
    if (j.hint) showToast(j.hint, "info", 4000);

    if (isEmergency) {
      setStatus("escalated", "Emergency escalation");
      $("msg").disabled     = true;
      $("sendBtn").disabled = true;
      if ($("micBtn")) $("micBtn").disabled = true;
    } else if (j.status === "error") {
      setStatus("error", "Error");
    } else if (j.phase === "done") {
      setStatus("done", "Intake complete");
      if ($("micBtn")) $("micBtn").disabled = true;
    } else {
      setStatus("active", "Session active");
    }

    if (j.phase === "done") await loadReport();

  } catch {
    showToast("Message failed. Please try again.", "error");
    setStatus("error", "Error");
  } finally {
    setTyping(false);
    if (!$("msg").disabled) $("msg").focus();
  }
}


/* ── Copy session ID ───────────────────────────────────── */
async function copyTid() {
  if (!threadId) return;
  await navigator.clipboard.writeText(threadId);
  showToast("Session ID copied.", "success");
}

/* ── Demo scenarios ────────────────────────────────────── */

/* ── Event bindings ────────────────────────────────────── */
window.addEventListener("DOMContentLoaded", () => {
  $("startBtn").addEventListener("click", start);
  $("sendBtn").addEventListener("click", sendMsg);
  $("msg").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMsg(); } });
  $("copyTidBtn").addEventListener("click", copyTid);

  // ── Voice wiring ────────────────────────────────────────
  // Push-to-talk: pointerdown starts, pointerup OR pointerleave stops.
  // Pointer events cover mouse + touch + pen with one handler set, and
  // pointerleave catches the case where the user drags off the button
  // before releasing — without it the recording would run until the
  // 30s ceiling.
  const mic = $("micBtn");
  if (mic) {
    const begin = e => { e.preventDefault(); startRecording(); };
    const end   = e => { e.preventDefault(); stopRecording(); };
    mic.addEventListener("pointerdown",  begin);
    mic.addEventListener("pointerup",    end);
    mic.addEventListener("pointerleave", end);
    mic.addEventListener("pointercancel", end);
    // Spacebar shortcut for desktop accessibility — only when the
    // composer input is not focused so we don't intercept normal typing.
    document.addEventListener("keydown", e => {
      if (e.code === "Space" && !mic.disabled && document.activeElement !== $("msg")
          && !mediaRecorder && voiceCfg.enabled) {
        e.preventDefault(); startRecording();
      }
    });
    document.addEventListener("keyup", e => {
      if (e.code === "Space" && mediaRecorder) { e.preventDefault(); stopRecording(); }
    });
  }

  $("ttsToggleBtn").addEventListener("click", toggleTts);
  $("vcSendBtn").addEventListener("click",   confirmVoiceSend);
  $("vcEditBtn").addEventListener("click",   editVoiceTranscript);
  $("vcCancelBtn").addEventListener("click", hideVoiceConfirm);

  const savedTid   = localStorage.getItem("threadId");
  const savedToken = localStorage.getItem("sessionToken");
  if (savedTid && savedToken) {
    sessionToken = savedToken;
    threadId     = savedTid;
    fetch(`/resume/${savedTid}`, {
      headers: { "Authorization": `Bearer ${savedToken}` },
    })
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(j => {
        $("tid").textContent = savedTid.slice(0, 12) + "…";
        $("sessionLabel").classList.remove("hidden");
        $("chat").innerHTML = "";
        $("msg").disabled     = false;
        $("sendBtn").disabled = false;
        currentPhase = j.phase || "identity";
        setStatus("active", "Session resumed");
        updatePhase("active", currentPhase);
        addMsg("assistant", j.reply);
        renderQuickReplies(j.quick_replies);
      })
      .catch(() => {
        localStorage.removeItem("threadId");
        localStorage.removeItem("sessionToken");
        sessionToken = null;
        threadId     = null;
      });
  }

  loadVoiceConfig();
});