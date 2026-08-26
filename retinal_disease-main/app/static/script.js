const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const fileInput = document.getElementById("fileInput");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const clearChatBtn = document.getElementById("clearChatBtn");
const dragOverlay = document.getElementById("dragOverlay");
const imageModal = document.getElementById("imageModal");
const modalImg = document.getElementById("modalImg");
const modalClose = document.getElementById("modalClose");
const appContainer = document.getElementById("appContainer");

let selectedFile = null;

function addBotMessage(html, extraClass = "") {
  const div = document.createElement("div");
  div.className = "msg bot" + (extraClass ? " " + extraClass : "");
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addUserImageMessage(file) {
  const div = document.createElement("div");
  div.className = "msg user";
  const img = document.createElement("img");
  img.src = URL.createObjectURL(file);
  div.appendChild(img);
  const caption = document.createElement("div");
  caption.textContent = "Eye photo bhej diya hai.";
  div.appendChild(caption);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function showTyping() {
  const div = document.createElement("div");
  div.className = "typing";
  div.id = "typingIndicator";
  div.innerHTML = "<span></span><span></span><span></span>";
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

function renderPredictions(preds) {
  return (
    '<div class="preds">' +
    preds
      .map(
        (p) => `
        <div class="pred-row">
          <span>${p.label}</span>
          <div class="pred-bar"><span style="width:${p.confidence}%"></span></div>
          <span>${p.confidence}%</span>
        </div>`
      )
      .join("") +
    "</div>"
  );
}

async function sendImage(file) {
  addUserImageMessage(file);
  showTyping();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    hideTyping();

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      addBotMessage(
        `<span class="error">Sorry, kuch gadbad ho gayi: ${err.detail || res.statusText}</span>`
      );
      return;
    }

    const data = await res.json();
    const replyHtml = data.reply.replace(/\n/g, "<br/>");
    const preds = data.looks_like_fundus ? renderPredictions(data.top_predictions) : "";
    const inferenceTime = data.inference_time ? `<span class="inference-time">⏱️ Processed in ${data.inference_time}s</span>` : "";
    const cls = data.looks_like_fundus ? "" : " warning";
    addBotMessage(replyHtml + preds + inferenceTime, cls);
  } catch (e) {
    hideTyping();
    addBotMessage('<span class="error">Server se connect nahi ho paaya. Kya backend chal raha hai?</span>');
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    selectedFile = fileInput.files[0];
    textInput.disabled = false;
    sendBtn.disabled = false;
    textInput.placeholder = `${selectedFile.name} — Send dabayein to analyze karne ke liye`;
  }
});

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (!selectedFile) return;
  sendImage(selectedFile);
  selectedFile = null;
  fileInput.value = "";
  textInput.value = "";
  textInput.disabled = true;
  sendBtn.disabled = true;
  textInput.placeholder = "Attach a photo of the eye to begin…";
});

function sendGreeting() {
  addBotMessage(
    "Hi! 👋 Main EyeCare Assistant hoon. Yeh sirf <b>fundus/retina-camera photos</b> ke liye kaam karta hai " +
      "(doctor ke special camera se li gayi, retina ke andar ki photo) — normal phone selfie ya eye close-up nahi.<br/><br/>" +
      "Aisi photo chahiye:<br/>" +
      '<img src="sample_fundus.jpg" alt="sample fundus photo" style="width:140px;border-radius:10px;margin-top:6px;" /><br/><br/>' +
      "📎 icon se aisi hi ek fundus photo attach karein."
  );
}

clearChatBtn.addEventListener("click", () => {
  chat.innerHTML = "";
  selectedFile = null;
  fileInput.value = "";
  textInput.value = "";
  textInput.disabled = true;
  sendBtn.disabled = true;
  textInput.placeholder = "Attach a photo of the eye to begin…";
  sendGreeting();
});

appContainer.addEventListener("dragover", (e) => {
  e.preventDefault();
  dragOverlay.classList.remove("hidden");
});

appContainer.addEventListener("dragleave", (e) => {
  e.preventDefault();
  if (e.relatedTarget && !appContainer.contains(e.relatedTarget)) {
    dragOverlay.classList.add("hidden");
  }
});

dragOverlay.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dragOverlay.classList.add("hidden");
});

appContainer.addEventListener("drop", (e) => {
  e.preventDefault();
  dragOverlay.classList.add("hidden");
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    const file = e.dataTransfer.files[0];
    if (file.type.startsWith("image/")) {
        selectedFile = file;
        textInput.disabled = false;
        sendBtn.disabled = false;
        textInput.placeholder = `${selectedFile.name} — Send dabayein to analyze karne ke liye`;
    }
  }
});

chat.addEventListener("click", (e) => {
  if (e.target.tagName === "IMG") {
    modalImg.src = e.target.src;
    imageModal.classList.remove("hidden");
  }
});

modalClose.addEventListener("click", () => {
  imageModal.classList.add("hidden");
});

imageModal.addEventListener("click", (e) => {
  if (e.target === imageModal) {
    imageModal.classList.add("hidden");
  }
});

sendGreeting();
