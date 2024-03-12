import execjs

from PIL import Image
import streamlit
from simple_latex_ocr.models import Latex_OCR

js_code = """
function encodeToBase64(latex) {
    var str = encodeURI(latex); // 进行URL编码
    str = btoa(str); // 进行Base64编码
    return str;
}
"""
ctx = execjs.compile(js_code)
model = Latex_OCR()
if __name__ == '__main__':
    streamlit.set_page_config(page_title='Simple-LaTeX-OCR')
    streamlit.title('Simple-LaTeX-OCR')
    streamlit.markdown(
        'Convert images of equations to corresponding LaTeX code.\n\nThis is based on the `Simple-LaTeX-OCR` module. Check it out (https://github.com/chaodreaming/Simple-LaTeX-OCR)')

    uploaded_file = streamlit.file_uploader(
        'Upload an image an equation',
        type=['png', 'jpg'],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        streamlit.image(image)
    else:
        streamlit.text('\n')

    if streamlit.button('Convert'):
        if uploaded_file is not None and image is not None:
            with streamlit.spinner('Computing'):
                response = model.predict(uploaded_file.getvalue())
            if "formula" in response:
                latex_code = response["formula"]
                latex_encode = ctx.call("encodeToBase64", latex_code)
                streamlit.code(latex_code, language='latex')
                streamlit.markdown(f'$\\displaystyle {latex_code}$')
                link_html = f'<a href="https://www.latexlive.com/#{latex_encode}" target="_blank"><button style="color: black;">Online Editing</button></a>'
                streamlit.markdown(link_html, unsafe_allow_html=True)
            else:
                streamlit.error(response)
        else:
            streamlit.error('Please upload an image.')
