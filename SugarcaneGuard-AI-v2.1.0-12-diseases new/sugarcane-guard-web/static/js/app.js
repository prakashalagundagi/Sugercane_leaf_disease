/* SugarcaneGuard AI — frontend logic */
const ORDER = ["Mosaic", "Smut", "Yellow Leaf", "Sett Rot", "Pokkah Boeng", "Banded Chlorosis", "Brown Spot", "Brown Rust", "Grassy Shoot", "Viral Disease", "Healthy Leaves", "Dried Leaves"];
let classInfo = {};
let currentFile = null;
let lastResult = null;
let activeSev = "all";

/* ---------------- health / mode badge ---------------- */
fetch("/api/health").then(r => r.json()).then(h => {
  const b = document.getElementById("modeBadge");
  const foot = document.getElementById("footMode");
  if (h.mode === "real") {
    b.textContent = "● live model";
    b.classList.add("real");
    foot.textContent = "Mode: live model (" + (h.model_file || "keras") + ")";
  } else {
    b.textContent = "● demo mode";
    foot.textContent = "Mode: demo (no trained model yet)";
    const cta = document.getElementById("trainCta");
    if (cta) cta.style.display = "block";
  }
}).catch(() => {});

/* ---------------- disease guide ---------------- */
fetch("/api/classes").then(r => r.json()).then(d => {
  classInfo = d;
  renderGuide();
});

document.getElementById("guideSearch").addEventListener("input", renderGuide);
document.querySelectorAll("#sevChips .chip").forEach(c => {
  c.addEventListener("click", () => {
    document.querySelectorAll("#sevChips .chip").forEach(x => x.classList.remove("active"));
    c.classList.add("active");
    activeSev = c.dataset.sev;
    renderGuide();
  });
});

function renderGuide() {
  const q = document.getElementById("guideSearch").value.trim().toLowerCase();
  const grid = document.getElementById("libGrid");
  const empty = document.getElementById("emptyMsg");
  grid.innerHTML = "";
  let shown = 0;
  ORDER.forEach(name => {
    const info = classInfo[name] || {};
    const sev = info.severity || "low";
    if (activeSev !== "all" && sev !== activeSev) return;
    const hay = (name + " " + (info.scientific || "") + " " + (info.symptoms || []).join(" ")).toLowerCase();
    if (q && !hay.includes(q)) return;
    shown++;
    const card = document.createElement("div");
    card.className = "lib-card";
    card.innerHTML = `
      <div class="top">
        <div class="icon">${info.icon || "🌿"}</div>
        <div><h3>${name}</h3><span class="sev ${sev}">${sev.toUpperCase()}${sev === "none" ? "" : " SEVERITY"}</span></div>
      </div>
      <div class="sci">${info.scientific || ""}</div>
      <details><summary>Symptoms &amp; management</summary>
        <ul><b>Symptoms:</b>${(info.symptoms || []).map(s => `<li>${s}</li>`).join("")}</ul>
        <ul style="margin-top:8px"><b>Management:</b>${(info.management || []).map(s => `<li>${s}</li>`).join("")}</ul>
      </details>`;
    grid.appendChild(card);
  });
  empty.style.display = shown ? "none" : "block";
}

/* ---------------- sample gallery ---------------- */
fetch("/static/samples.json").then(r => r.json()).then(list => {
  const box = document.getElementById("samples");
  list.forEach(s => {
    const el = document.createElement("div");
    el.className = "sample";
    el.innerHTML = `<img src="/static/${s.file}" alt="${s.label}"><span>${s.label}</span>`;
    el.onclick = () => loadSample("/static/" + s.file, s.label);
    box.appendChild(el);
  });
}).catch(() => {});

async function loadSample(url, label) {
  const blob = await (await fetch(url)).blob();
  setFile(new File([blob], label + ".jpg", { type: blob.type }));
  document.getElementById("scanner").scrollIntoView({ behavior: "smooth" });
}

/* ---------------- upload flow ---------------- */
const drop = document.getElementById("dropZone"), fi = document.getElementById("fileInput");
drop.onclick = () => fi.click();
["dragover", "dragenter"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", ev => { if (ev.dataTransfer.files[0]) setFile(ev.dataTransfer.files[0]); });
fi.onchange = () => { if (fi.files[0]) setFile(fi.files[0]); };

function setFile(f) {
  if (!f.type.startsWith("image/")) { alert("Please choose an image file."); return; }
  if (f.size > 10 * 1024 * 1024) { alert("That image is larger than 10 MB."); return; }
  currentFile = f;
  document.getElementById("previewImg").src = URL.createObjectURL(f);
  document.getElementById("stage").classList.add("show");
  document.getElementById("result").classList.remove("show");
}

document.getElementById("clearBtn").onclick = resetScan;
document.getElementById("newScanBtn").onclick = () => {
  resetScan();
  document.getElementById("scanner").scrollIntoView({ behavior: "smooth" });
};
function resetScan() {
  currentFile = null; fi.value = ""; lastResult = null;
  document.getElementById("stage").classList.remove("show");
  document.getElementById("result").classList.remove("show");
}

document.getElementById("analyzeBtn").onclick = async () => {
  if (!currentFile) return;
  const sp = document.getElementById("spinner");
  sp.style.display = "block";
  document.getElementById("result").classList.remove("show");
  try {
    const fd = new FormData(); fd.append("image", currentFile);
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || "Prediction failed"); }
    lastResult = await r.json();
    renderResult(lastResult);
  } catch (err) { alert("Couldn't analyze that image: " + err.message); }
  finally { sp.style.display = "none"; }
};

/* ---------------- result dashboard ---------------- */
const RING_C = 351.9;
function renderResult(d) {
  document.getElementById("rClass").textContent = d.predicted_class;
  const info = d.top3[0].info || {};
  document.getElementById("rSci").textContent = info.scientific || "";
  const pct = Math.round(d.confidence * 100);
  document.getElementById("rPct").textContent = pct + "%";
  const arc = document.getElementById("ringArc");
  arc.style.transition = "none"; arc.style.strokeDashoffset = RING_C;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    arc.style.transition = "stroke-dashoffset 1.1s cubic-bezier(.2,.8,.2,1)";
    arc.style.strokeDashoffset = (RING_C * (1 - d.confidence)).toFixed(1);
  }));
  const sev = document.getElementById("rSev");
  sev.className = "sev " + (info.severity || "low");
  sev.textContent = (info.severity || "").toUpperCase() + (info.severity === "none" ? "" : " SEVERITY");
  document.getElementById("rMode").textContent = d.mode === "real"
    ? "● live model" + (d.model_file ? " · " + d.model_file : "")
    : "● demo prediction";

  const bars = document.getElementById("bars"); bars.innerHTML = "";
  d.top3.forEach((t, i) => {
    const row = document.createElement("div"); row.className = "bar-row";
    row.innerHTML = `<div class="lbl"><span>${i + 1}. ${t.class}</span><span>${Math.round(t.confidence * 100)}%</span></div>
      <div class="bar"><i data-w="${Math.round(t.confidence * 100)}"></i></div>`;
    bars.appendChild(row);
  });
  requestAnimationFrame(() => requestAnimationFrame(() => {
    bars.querySelectorAll("i").forEach(el => el.style.width = el.dataset.w + "%");
  }));

  document.getElementById("rSymptoms").innerHTML = (info.symptoms || []).map(s => `<li>${s}</li>`).join("");
  document.getElementById("rManage").innerHTML = (info.management || []).map(s => `<li>${s}</li>`).join("");

  const note = document.getElementById("demoNote");
  if (d.mode === "demo") { note.style.display = "block"; note.textContent = "⚠️ " + (d.note || "Demo mode."); }
  else note.style.display = "none";

  document.getElementById("inferTime").textContent =
    d.inference_ms != null ? `⚡ Analyzed in ${(d.inference_ms/1000).toFixed(2)}s` : "";

  buildPrintReport(d, info);
  document.getElementById("result").classList.add("show");
  document.getElementById("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* tabs */
document.querySelectorAll(".tabbar button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabbar button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tabpane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

/* ---------------- printable report ---------------- */
function buildPrintReport(d, info) {
  const now = new Date().toLocaleString();
  const rows = d.top3.map((t, i) =>
    `<tr><td>${i + 1}</td><td>${t.class}</td><td>${(t.confidence * 100).toFixed(2)}%</td></tr>`).join("");
  document.getElementById("print-report").innerHTML = `
    <div class="pr-head">
      <h1>🌾 SugarcaneGuard AI — Diagnosis Report</h1>
      <div>Generated: ${now} &nbsp;·&nbsp; Mode: ${d.mode === "real" ? "Live model" + (d.model_file ? " (" + d.model_file + ")" : "") : "DEMO (simulated)"}</div>
    </div>
    <h2>Diagnosis</h2>
    <table>
      <tr><th>Predicted condition</th><td><b>${d.predicted_class}</b></td></tr>
      <tr><th>Confidence</th><td>${(d.confidence * 100).toFixed(2)}%</td></tr>
      <tr><th>Scientific name</th><td>${info.scientific || "—"}</td></tr>
      <tr><th>Severity</th><td>${(info.severity || "—").toUpperCase()}</td></tr>
    </table>
    <h2>Top predictions</h2>
    <table><tr><th>#</th><th>Condition</th><th>Confidence</th></tr>${rows}</table>
    <h2>Symptoms to verify</h2>
    <ul>${(info.symptoms || []).map(s => `<li>${s}</li>`).join("")}</ul>
    <h2>Recommended actions</h2>
    <ul>${(info.management || []).map(s => `<li>${s}</li>`).join("")}</ul>
    <div class="pr-foot">⚠️ This report supports field scouting and is not a substitute for professional agronomic advice. Confirm serious outbreaks with a qualified agronomist.</div>`;
}
document.getElementById("printBtn").onclick = () => { if (lastResult) window.print(); };

/* ---------------- FAQ accordion ---------------- */
document.querySelectorAll(".faq-item").forEach(item => {
  const q = item.querySelector(".faq-q"), a = item.querySelector(".faq-a");
  q.addEventListener("click", () => {
    const open = item.classList.contains("open");
    document.querySelectorAll(".faq-item.open").forEach(o => {
      o.classList.remove("open"); o.querySelector(".faq-a").style.maxHeight = null;
    });
    if (!open) { item.classList.add("open"); a.style.maxHeight = a.scrollHeight + "px"; }
  });
});
