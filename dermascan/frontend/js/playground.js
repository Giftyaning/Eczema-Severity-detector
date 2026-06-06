/* Playground page: live tester for POST /predict. */

(function () {
  const input = document.getElementById("fileInput");
  const dzText = document.getElementById("dzText");
  const thumb = document.getElementById("thumb");
  const sendBtn = document.getElementById("sendBtn");
  const spin = document.getElementById("spin");
  const err = document.getElementById("err");
  const responseEl = document.getElementById("response");
  const statusEl = document.getElementById("status");
  const overlay = document.getElementById("overlay");
  const dz = document.getElementById("dropzone");
  let selected = null;

  function setFile(file) {
    if (!file) return;
    if (!["image/jpeg", "image/png"].includes(file.type)) {
      err.textContent = "Please choose a JPEG or PNG image.";
      return;
    }
    err.textContent = "";
    selected = file;
    thumb.src = URL.createObjectURL(file);
    thumb.classList.remove("hidden");
    dzText.innerHTML = `<strong>${file.name}</strong> · ${(file.size / 1024).toFixed(0)} KB`;
    sendBtn.disabled = false;
  }

  input.addEventListener("change", (e) => setFile(e.target.files[0]));
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); })
  );
  dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  sendBtn.addEventListener("click", async () => {
    if (!selected) return;
    err.textContent = "";
    statusEl.textContent = "";
    overlay.classList.add("hidden");
    spin.classList.remove("hidden");
    sendBtn.disabled = true;

    const model = document.getElementById("model").value;
    const gradcam = document.getElementById("gradcam").value === "true";

    const t0 = performance.now();
    try {
      const data = await fetchPredict(selected, { model, gradcam });
      const ms = Math.round(performance.now() - t0);
      statusEl.innerHTML = `<span style="color:var(--teal)">200 OK</span> · ${ms} ms round-trip`;
      responseEl.innerHTML = highlightJSON(data);
      if (data.gradcam_base64) {
        overlay.src = `data:image/png;base64,${data.gradcam_base64}`;
        overlay.classList.remove("hidden");
      }
    } catch (e) {
      statusEl.innerHTML = `<span style="color:var(--coral)">error</span>`;
      responseEl.innerHTML = highlightJSON({ status: "error", detail: e.message });
      err.textContent = e.message;
    } finally {
      spin.classList.add("hidden");
      sendBtn.disabled = false;
    }
  });
})();
