from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "Nici o imagine încărcată"
    file = request.files['image']
    if file.filename == '':
        return "Nici o imagine selectată"

    format_option = request.form.get('format', 'default')

    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return "Fișierul încărcat nu este o imagine validă."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if format_option == 'default':
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 7)
        processed = thresh
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)
        processed = cv2.dilate(processed, kernel, iterations=1)

    elif format_option == 'background':
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 13)
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        processed = cv2.medianBlur(processed, 5)

    elif format_option == 'visible_lines':
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 13)
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        processed = cv2.dilate(processed, kernel, iterations=2)

    else:
        return "Formatul selectat nu este recunoscut."

    _, buffer = cv2.imencode('.png', processed)
    return send_file(io.BytesIO(buffer), mimetype='image/png', as_attachment=True, download_name='coloring_page.png')

if __name__ == '__main__':
    app.run(debug=True)
