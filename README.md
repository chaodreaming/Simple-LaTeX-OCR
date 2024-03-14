# Simple-LaTeX-OCR
### Performance
| BLEU score | normed edit distance | token accuracy |
|------------|----------------------|----------------|
| 0.92       | 0.05                 | 0.75           |
## Online experience

### https://huggingface.co/spaces/chaodreaming/Simple-LaTeX-OCR

### https://www.modelscope.cn/studios/chaodreaming/Simple-LaTeX-OCR/summary

 Install the package `simple_latex_ocr`: 

```
pip install simple_latex_ocr
```

Use from within Python

    
    from simple_latex_ocr.models import Latex_OCR
    model = Latex_OCR()
    img_path = "tests/test_files/5.png"
    result = model.predict(img_path)
    print(result['formula'])
    print(result['confidence'])
    print(result['elapse'])

#### Used by command line
```bash
$ simple_latex_ocr tests/test_files/2.png
```


#### Used by api
```bash
$ python -m simple_latex_ocr.api.run

#You can use test.py to initiate a request for verification
```

Using streamlit has a nice interface, but you can only select files locally.
```
streamlit run streamlit_app.py
```
![img.png](img.png)
Using flask you can copy the formula directly

```commandline
python flask_app.py
```


## Contribution
Contributions of any kind are welcome.

## Acknowledgment
Code taken and modified from [lukas-blecher](https://github.com/lukas-blecher/LaTeX-OCR), [RapidAI](https://github.com/RapidAI/RapidLaTeXOCR),[ultralytics](https://github.com/ultralytics/ultralytics)
