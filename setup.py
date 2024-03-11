from setuptools import setup, find_packages

setup(
    name='simplelatexocr',
    version='0.0.3',
    author='chaodreaming',  # 修改为你的名字或者组织名称
    author_email='chaodreaming@gmail.com',  # 修改为你的邮箱地址
    description='A simple LaTeX OCR package',  # 项目的简短描述
    long_description=open('README.md').read(),  # 可以把 README.md 作为长描述
    long_description_content_type='text/markdown',  # 设置长描述的格式为 markdown
    license='Apache-2.0',  # 项目的开源协议
    packages=find_packages(),  # 自动找到项目中的所有包
    install_requires=[
        'onnxruntime-gpu',
        'tokenizers>=0.13.2',
        'numpy',
        'opencv-python',
        'Pillow>=9.2.0',
        'PyYAML',
        'transformers',
        'flask',
        'fastapi',
        'uvicorn'
    ],
    python_requires='>=3.8',  # 你的包支持的最小Python版本
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
'''
python setup.py bdist_wheel
python setup.py sdist
twine upload dist/*
'''
