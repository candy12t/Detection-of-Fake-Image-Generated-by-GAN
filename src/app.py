from flask import Flask, redirect, render_template, request, url_for

from predict_image import predict_image

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        if not request.files["image"]:
            return redirect(url_for("index"))

        img = request.files["image"].stream
        predict, proba = predict_image(img)
        return render_template("index.html", predict=predict, proba=proba)


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
