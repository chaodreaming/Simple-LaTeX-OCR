from simplelatexocr.models import Latex_OCR

detect_path = 'models/best.onnx'
encoder_path = 'models/encoder.onnx'
decoder_path = 'models/decoder.onnx'
tokenizer_json = 'models/tokenizer.json'
model = Latex_OCR(
        detect_path=detect_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        tokenizer_json=tokenizer_json,
    )


img_path = "tests/test_files/5.png"
with open(img_path, "rb") as f:
    data = f. read()
result = model.predict(img_path)
print(result)
# 检测耗时： 0.8744056224822998
# 置信度为:93.11%，
# 公式为:
# \frac{x^{2}}{a^{2}}-\frac{y^{2}}{b^{2}}=\mathsf{1}
# {'formula': '\\frac{x^{2}}{a^{2}}-\\frac{y^{2}}{b^{2}}=\\mathsf{1}', 'confidence': '97.21%', 'elapse': '1,897ms'}