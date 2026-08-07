// ==========================================
// LipVision
// Main JavaScript
// ==========================================

document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.querySelector('input[type="file"]');

  const uploadBox = document.querySelector(".upload-box");

  if (!fileInput || !uploadBox) {
    return;
  }

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length === 0) {
      return;
    }

    const file = fileInput.files[0];

    uploadBox.querySelector("h3").textContent = file.name;

    uploadBox.querySelector("p").textContent =
      `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  });
});
