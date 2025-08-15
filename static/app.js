async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    const models = Array.isArray(data) ? data : data.models || [];
    const select = document.getElementById("modelSelect");
    select.innerHTML = "";

    // Filter client-side too (safety net)
    const chatModels = models.filter(m => (m?.name || "").toLowerCase().indexOf("embed") === -1);

    if (chatModels.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "No models found (pull llama3)";
      opt.disabled = true;
      select.appendChild(opt);
      return;
    }

    chatModels.forEach(m => {
      const name = m && m.name ? m.name : String(m);
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    });

    // Prefer llama3 if present
    const preferred = Array.from(select.options).find(o => o.value.startsWith("llama3"));
    if (preferred) select.value = preferred.value;
  } catch (err) {
    console.error("Model load failed:", err);
  }
}

async function uploadPDF() {
  const file = document.getElementById("pdfFile").files[0];
  const status = document.getElementById("uploadStatus");
  if (!file) return alert("Please select a PDF file first.");

  const form = new FormData();
  form.append("file", file);

  status.textContent = "Uploading & indexing...";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const json = await res.json();
    status.textContent = json.message || json.error || "Done.";
  } catch (e) {
    status.textContent = "Upload failed.";
  }
}

async function askQuestion() {
  const question = document.getElementById("questionInput").value.trim();
  const model = document.getElementById("modelSelect").value;
  const temperature = parseFloat(document.getElementById("temperatureRange").value);
  const answerBox = document.getElementById("answerBox");

  if (!question) return alert("Please type a question.");

  answerBox.textContent = "Thinking…";
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, model, temperature })
    });
    const json = await res.json();
    answerBox.textContent = json.answer || json.error || "(no response)";
  } catch (e) {
    answerBox.textContent = "Request failed.";
  }
}

document.getElementById("uploadBtn").addEventListener("click", uploadPDF);
document.getElementById("askBtn").addEventListener("click", askQuestion);
document.getElementById("temperatureRange").addEventListener("input", (e) => {
  document.getElementById("temperatureValue").textContent = e.target.value;
});

window.addEventListener("DOMContentLoaded", loadModels);
