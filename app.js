(function(){
// Converts 21 MediaPipe hand landmarks into a normalized feature vector
// (translation- and scale-invariant). Shared by all predictors.
function normalize(landmarks) {
  const w = landmarks[0];
  const pts = landmarks.map(p => [p.x - w.x, p.y - w.y, (p.z || 0) - (w.z || 0)]);
  const s = Math.hypot(...pts[9]) || 1; // wrist → middle MCP
  return pts.flat().map(v => v / s);
}

// Finger extension states: [thumb, index, middle, ring, pinky]
function fingerStates(lm) {
  const d = (a, b) => Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y);
  const ext = (tip, pip) => d(tip, 0) > d(pip, 0) * 1.15;
  const thumb = d(4, 17) > d(3, 17) * 1.1 && d(4, 5) > d(3, 5) * 0.9;
  return [thumb, ext(8, 6), ext(12, 10), ext(16, 14), ext(20, 18)];
}

/**
 * Predictor interface — every engine implements:
 *   async predict(hands: Array<{landmarks, handedness}>) => { label, confidence } | null
 * Swap engines in app.js. To integrate an ML model, implement ApiPredictor or TfjsPredictor.
 */

// 1) Built-in rule engine based on finger geometry
const RULES = [
  { label: "HELLO", desc: "Open palm", f: [1, 1, 1, 1, 1] },
  { label: "YES", desc: "Closed fist", f: [0, 0, 0, 0, 0] },
  { label: "GOOD", desc: "Thumb up", f: [1, 0, 0, 0, 0] },
  { label: "ONE", desc: "Index finger", f: [0, 1, 0, 0, 0] },
  { label: "PEACE", desc: "V sign", f: [0, 1, 1, 0, 0] },
  { label: "THREE", desc: "Three fingers", f: [0, 1, 1, 1, 0] },
  { label: "I LOVE YOU", desc: "Thumb, index, pinky", f: [1, 1, 0, 0, 1] },
  { label: "CALL", desc: "Thumb + pinky", f: [1, 0, 0, 0, 1] },
  { label: "FOUR", desc: "Four fingers", f: [0, 1, 1, 1, 1] },
  { label: "OK", desc: "Thumb-index circle", ok: true },
];

class RulePredictor {
  name = "rules";
  async predict(hands) {
    if (!hands.length) return null;
    const lm = hands[0].landmarks;
    const okDist = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) / (Math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) || 1);
    const s = fingerStates(lm);
    if (okDist < 0.25 && s[2] && s[3]) return { label: "OK", confidence: Math.max(0.6, 1 - okDist * 2) };
    let best = null;
    for (const r of RULES) {
      if (!r.f) continue;
      const match = r.f.reduce((a, v, i) => a + (Boolean(v) === s[i] ? 1 : 0), 0) / 5;
      if (!best || match > best.confidence) best = { label: r.label, confidence: match };
    }
    if (hands.length === 2 && best.label === "HELLO") best = { label: "NAMASTE?", confidence: 0.55 };
    return best.confidence >= 0.8 ? best : { label: best.label, confidence: best.confidence * 0.6 };
  }
}

// 2) k-nearest-neighbour model trained from collected samples (runs on-device)
class KnnPredictor {
  name = "custom kNN";
  constructor(samples, k = 5) { this.samples = samples; this.k = k; }
  async predict(hands) {
    if (!hands.length || this.samples.length < 2) return null;
    const v = normalize(hands[0].landmarks);
    const dists = this.samples.map(s => ({ label: s.label, d: Math.hypot(...s.vector.map((x, i) => x - v[i])) }))
      .sort((a, b) => a.d - b.d).slice(0, this.k);
    const votes = {};
    dists.forEach(x => (votes[x.label] = (votes[x.label] || 0) + 1));
    const [label, n] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
    const closeness = Math.max(0, 1 - dists[0].d / 3);
    return { label, confidence: (n / dists.length) * 0.7 + closeness * 0.3 };
  }
}

// 3) Remote ML API — send landmarks to your backend model
class ApiPredictor {
  name = "api";
  constructor(url, apiKey) { this.url = url; this.apiKey = apiKey; this.busy = false; this.last = null; }
  async predict(hands) {
    if (!hands.length) return null;
    if (this.busy) return this.last; // throttle: one request in flight
    this.busy = true;
    try {
      const res = await fetch(this.url, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}) },
        // Expected response: { "label": "HELLO", "confidence": 0.93 }
        body: JSON.stringify({ hands: hands.map(h => ({ handedness: h.handedness, landmarks: h.landmarks, vector: normalize(h.landmarks) })) }),
      });
      this.last = res.ok ? await res.json() : null;
    } catch { this.last = null; } finally { this.busy = false; }
    return this.last;
  }
}

// 4) TensorFlow.js model stub — load a model exported from your training pipeline
class TfjsPredictor {
  name = "tfjs";
  constructor(modelUrl, labels) { this.modelUrl = modelUrl; this.labels = labels; }
  async load() { this.model = await window.tf.loadLayersModel(this.modelUrl); }
  async predict(hands) {
    if (!this.model || !hands.length) return null;
    const out = this.model.predict(window.tf.tensor2d([normalize(hands[0].landmarks)]));
    const p = await out.data(); out.dispose();
    const i = p.indexOf(Math.max(...p));
    return { label: this.labels[i], confidence: p[i] };
  }
}

const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const STABLE_FRAMES = 12, MIN_CONF = 0.75;

const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
let landmarker, stream, raf, lastTs = -1, frames = 0, fpsT = performance.now();
let latestHands = [], streak = { label: null, n: 0 }, lastCommitted = null;
let history = [], recording = null;
let samples = JSON.parse(localStorage.getItem("isl-samples") || "[]");

// ---- Engine selection: replace with ApiPredictor / TfjsPredictor to plug in your model ----
const rules = new RulePredictor();
const engine = () => (new Set(samples.map(s => s.label)).size >= 2 ? new KnnPredictor(samples) : rules);

function setStatus(text, cls = "") { $("statusText").textContent = text; $("status").className = "status " + cls; }

$("legend").innerHTML = RULES.map(r => `<li><b>${r.label}</b><br>${r.desc}</li>`).join("");

async function loadLandmarker() {
  if (landmarker) return;
  setStatus("Loading hand model…");
  const { FilesetResolver, HandLandmarker } = await import(MP);
  const fs = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
  const opts = d => ({ baseOptions: { modelAssetPath: MODEL, delegate: d }, runningMode: "VIDEO", numHands: 2 });
  try { landmarker = await HandLandmarker.createFromOptions(fs, opts("GPU")); }
  catch { landmarker = await HandLandmarker.createFromOptions(fs, opts("CPU")); }
}

async function start() {
  $("startBtn").disabled = true;
  try {
    await loadLandmarker();
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" }, audio: false });
    video.srcObject = stream; await video.play();
    $("placeholder").hidden = true; $("stopBtn").disabled = false;
    setStatus("Live", "live"); loop();
  } catch (e) {
    console.error(e); $("startBtn").disabled = false;
    setStatus(e.name === "NotAllowedError" ? "Camera permission denied" : "Could not start camera", "err");
  }
}

function stop() {
  cancelAnimationFrame(raf); stream?.getTracks().forEach(t => t.stop()); stream = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("placeholder").hidden = false; $("startBtn").disabled = false; $("stopBtn").disabled = true;
  $("caption").classList.remove("on"); setStatus("Camera off");
}

async function loop() {
  if (!stream) return;
  if (video.currentTime !== lastTs && video.videoWidth) {
    lastTs = video.currentTime;
    const r = landmarker.detectForVideo(video, performance.now());
    latestHands = (r.landmarks || []).map((l, i) => ({ landmarks: l, handedness: r.handedness?.[i]?.[0]?.categoryName }));
    draw(latestHands);
    if (recording) collect();
    const eng = engine();
    $("engine").textContent = "Engine: " + eng.name;
    show(await eng.predict(latestHands));
    frames++;
    const now = performance.now();
    if (now - fpsT > 1000) { $("fps").textContent = `${frames} fps`; frames = 0; fpsT = now; }
  }
  raf = requestAnimationFrame(loop);
}

function draw(hands) {
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!$("showLandmarks").checked) return;
  const W = canvas.width, H = canvas.height;
  hands.forEach((h, hi) => {
    ctx.strokeStyle = hi ? "#5fd39a" : "#f4a237"; ctx.lineWidth = 4;
    CONNECTIONS.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(h.landmarks[a].x * W, h.landmarks[a].y * H); ctx.lineTo(h.landmarks[b].x * W, h.landmarks[b].y * H); ctx.stroke(); });
    ctx.fillStyle = "#fff";
    h.landmarks.forEach(p => { ctx.beginPath(); ctx.arc(p.x * W, p.y * H, 5, 0, 7); ctx.fill(); });
  });
}

function show(pred) {
  const c = pred ? Math.round(pred.confidence * 100) : 0;
  $("word").textContent = pred ? pred.label : "—";
  $("conf").textContent = c + "%"; $("bar").style.width = c + "%"; $("meter").setAttribute("aria-valuenow", c);
  if (pred && pred.confidence >= MIN_CONF) {
    streak = streak.label === pred.label ? { label: pred.label, n: streak.n + 1 } : { label: pred.label, n: 1 };
    if (streak.n === STABLE_FRAMES && pred.label !== lastCommitted) commit(pred);
  } else if (!pred) { streak = { label: null, n: 0 }; lastCommitted = null; }
}

function commit(pred) {
  lastCommitted = pred.label;
  history.unshift({ ...pred, time: new Date() }); history = history.slice(0, 50);
  $("captionWord").textContent = pred.label; $("caption").classList.add("on");
  clearTimeout(commit.t); commit.t = setTimeout(() => $("caption").classList.remove("on"), 1500);
  if ($("speak").checked && "speechSynthesis" in window) speechSynthesis.speak(Object.assign(new SpeechSynthesisUtterance(pred.label.toLowerCase()), { lang: "en-IN" }));
  renderHistory();
}

function renderHistory() {
  $("sentence").textContent = history.length ? [...history].reverse().map(h => h.label).join(" ") : "Your sentence will appear here.";
  $("history").innerHTML = history.map(h => `<li><b>${h.label}</b><span>${Math.round(h.confidence * 100)}% · ${h.time.toLocaleTimeString()}</span></li>`).join("");
}

// ---- Sample collection ----
function collect() {
  if (!latestHands.length) return;
  samples.push({ label: recording.label, vector: normalize(latestHands[0].landmarks) });
  if (--recording.left <= 0) { recording = null; $("recordBtn").textContent = "● Record 30"; saveSamples(); }
  else $("recordBtn").textContent = `Recording… ${recording.left}`;
}
function saveSamples() {
  localStorage.setItem("isl-samples", JSON.stringify(samples));
  const counts = samples.reduce((a, s) => ((a[s.label] = (a[s.label] || 0) + 1), a), {});
  $("samples").innerHTML = Object.entries(counts).map(([l, n]) => `<li>${l} · ${n}</li>`).join("") || "<li>No samples yet</li>";
}

$("recordBtn").onclick = () => {
  const label = $("label").value.trim().toUpperCase();
  if (!label) return $("label").focus();
  if (!stream) return setStatus("Start the camera first", "err");
  recording = { label, left: 30 };
};
$("exportBtn").onclick = () => {
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([JSON.stringify(samples)], { type: "application/json" })), download: "isl-samples.json" });
  a.click();
};
$("importFile").onchange = async e => { const f = e.target.files[0]; if (f) { samples = samples.concat(JSON.parse(await f.text())); saveSamples(); } };
$("resetSamples").onclick = () => { if (confirm("Delete all collected samples?")) { samples = []; saveSamples(); } };
$("startBtn").onclick = start;
$("stopBtn").onclick = stop;
$("clearBtn").onclick = () => { history = []; lastCommitted = null; renderHistory(); };
$("copyBtn").onclick = () => navigator.clipboard?.writeText($("sentence").textContent);
$("mirror").onchange = e => $("videoWrap").classList.toggle("mirror", e.target.checked);
$("videoWrap").classList.add("mirror");
saveSamples();
})();
