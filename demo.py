from simplelatexocr.models import Latex_OCR
model = Latex_OCR()
img_path = "tests/test_files/5.png"
result = model.predict(img_path)
print(result['formula'])
print(result['confidence'])
print(result['elapse'])
