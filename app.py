import requests
from PIL import Image
import streamlit
from simple_latex_ocr.models import Latex_OCR
model=Latex_OCR()
if __name__ == '__main__':
    streamlit.set_page_config(page_title='Simple-LaTeX-OCR')
    streamlit.title('Simple-LaTeX-OCR')
    streamlit.markdown('Convert images of equations to corresponding LaTeX code.\n\nThis is based on the `Simple-LaTeX-OCR` module. Check it out (https://github.com/chaodreaming/Simple-LaTeX-OCR)')

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

                response =model.predict(uploaded_file.getvalue())
            if "formula" in response:
                latex_code = response["formula"]
                streamlit.code(latex_code, language='latex')
                streamlit.markdown(f'$\\displaystyle {latex_code}$')
                # 添加跳转按钮
                streamlit.markdown(
                    '<a href="https://www.latexlive.com/" target="_blank"><button style="color: black;">Online Editing</button></a>',
                    unsafe_allow_html=True)
            else:
                streamlit.error(response)
        else:
            streamlit.error('Please upload an image.')