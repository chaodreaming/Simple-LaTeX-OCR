from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
from simple_latex_ocr.models import Latex_OCR
model=Latex_OCR()
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    global  model
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 假设这里是你的OCR模型处理图片并返回LaTeX
        latex_result = model.predict(filepath)
        print(latex_result)

        return jsonify({'latex': latex_result["formula"]})
    else:
        return jsonify({'error': 'File not allowed'}), 400


if __name__ == '__main__':
    app.run(port=9000,debug=True)