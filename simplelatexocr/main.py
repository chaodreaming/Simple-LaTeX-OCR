from simplelatexocr.models import Latex_OCR


model = Latex_OCR()


img_path = "../tests/test_files/5.png"
with open(img_path, "rb") as f:
    data = f. read()
result = model.predict(img_path)
print(result['formula'])
print(result['confidence'])
print(result['elapse'])
