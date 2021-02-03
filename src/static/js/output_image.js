const localImg = localStorage.getItem("imgData");
const img = document.createElement("img");
img.src = localImg;
preview.appendChild(img);
