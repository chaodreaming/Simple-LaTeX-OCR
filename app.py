import json
from flask import Flask
from flask import request


from simplelatexocr.models import Latex_OCR
REQUEST_ID_HEADER = 'x-fc-request-id'

detect_path="models/best.onnx"
encoder_path = "models/encoder.onnx"
decoder_path = "models/decoder.onnx"
tokenizer_json = "models/tokenizer.json"

model = Latex_OCR(
        detect_path=detect_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        tokenizer_json=tokenizer_json,
    )

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST' and 'image' in request.files:
        image = request.files['image']
        res = model.predict(image)
        print("formula: ", res["formula"])
        print("confidence: ", res["confidence"])
        print("elapse: ", res["elapse"])
        print(res)
        return json.dumps(res)
    else:
        return json.dumps({"error": "info erro"})

if __name__ == '__main__':

    app.run(host='0.0.0.0',port=9000,debug=False)

