from simplelatexocr.models import Latex_OCR


model = Latex_OCR()


img_path = "../tests/test_files/5.png"
with open(img_path, "rb") as f:
    data = f. read()
result = model.predict(img_path)
print(result)
# 检测耗时： 0.8744056224822998
# 置信度为:93.11%，
# 公式为:
# \frac{x^{2}}{a^{2}}-\frac{y^{2}}{b^{2}}=\mathsf{1}
# {'formula': '\\frac{x^{2}}{a^{2}}-\\frac{y^{2}}{b^{2}}=\\mathsf{1}', 'confidence': '97.21%', 'elapse': '1,897ms'}