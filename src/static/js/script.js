const previewFile = (file) => {
  const preview = document.getElementById("preview");
  const reader = new FileReader();

  reader.onload = (e) => {
    const imgeUrl = e.target.result;
    const img = document.createElement("img");
    img.id = "preview_image"
    img.src = imgeUrl;
    preview.appendChild(img);
  }

  reader.readAsDataURL(file);
}

const fileInput = document.getElementById("input_image");
const handleFileSelect = () => {
  const files = fileInput.files
  previewFile(files[0])
}

const removeResult = () => {
  const result = document.getElementById("result");
  result.textContent = "";
}

const removeImage = () => {
  const preview = document.getElementById("preview");
  const preview_iamge = document.getElementById("preview_image");
  preview.removeChild(preview_iamge);
}

fileInput.addEventListener("change", handleFileSelect, false);
fileInput.addEventListener("click", removeResult, false)
fileInput.addEventListener("change", removeImage, false);
