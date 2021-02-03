const preview = document.getElementById("preview");
const fileInput = document.getElementById("input_image");

const handleFileSelect = () => {
  if(preview.childElementCount) {
    const child = preview.childNodes[0];
    preview.removeChild(child);
  }
  const result = document.getElementById("result");
  if(result && result.textContent) {
    result.textContent = "";
  }
  const files = fileInput.files;
  previewFile(files[0]);
}

const previewFile = (file) => {
  const img = document.createElement("img");
  const reader = new FileReader();
  reader.onload = (e) => {
    img.id = "preview_image";
    img.src = e.target.result;
    localStorage.setItem("imgData", img.src);
  }
  preview.appendChild(img);
  reader.readAsDataURL(file);
}

fileInput.addEventListener("change", handleFileSelect);
