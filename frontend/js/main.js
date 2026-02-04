const input = document.getElementById("messageInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultBox = document.getElementById("result");
const classificationText = document.getElementById("classification");
const confidenceText = document.getElementById("confidence");

const exampleButtons = document.querySelectorAll(".example");

// 👉 NÜMUNƏ MESAJLAR
exampleButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    input.value = btn.textContent.trim();
    input.focus(); // UX bonus
  });
});

analyzeBtn.addEventListener("click", async () => {
  const text = input.value.trim();
  if (!text) return;

  resultBox.classList.remove("hidden");
  classificationText.textContent = "Analyzing...";
  confidenceText.textContent = "";

  try {
    const res = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!res.ok) throw new Error("API error");

    const data = await res.json();
    renderResult(data);

  } catch {
    classificationText.textContent = "ERROR";
    confidenceText.textContent = "Backend unreachable";
  }
});

function renderResult({ label, confidence }) {
  resultBox.classList.remove("spam", "safe");
  resultBox.classList.add(label === "SPAM" ? "spam" : "safe");

  classificationText.textContent =
    label === "SPAM" ? "SPAM DETECTED" : "SAFE MESSAGE";

  confidenceText.textContent = `Confidence: ${confidence}%`;
}
