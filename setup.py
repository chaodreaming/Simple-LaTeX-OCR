#!/usr/bin/env python
from pathlib import Path
from typing import List

from setuptools import setup


def read_txt(txt_path: str) -> List:
    if not isinstance(txt_path, str):
        txt_path = str(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        data = list(map(lambda x: x.rstrip("\n"), f))
    return data


def get_readme() -> str:
    root_dir = Path(__file__).resolve().parent
    readme_path = str(root_dir / "docs" / "doc_whl.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    return readme




MODULE_NAME = "simple_latex_ocr"

setup(
    name=MODULE_NAME,
    version="0.0.8",
    author='chaodreaming',  # 修改为你的名字或者组织名称
    author_email='chaodreaming@gmail.com',  # 修改为你的邮箱地址
    platforms="Any",
    url="https://github.com/chaodreaming/Simple-LaTeX-OCR",
    include_package_data=True,
    install_requires=read_txt("requirements.txt"),
    packages=[MODULE_NAME],
    package_data={"": ["*.yaml"]},
    keywords=["ocr, image to text, latex"],
    description='A simple LaTeX OCR package',  # 项目的简短描述
    long_description=open('README.md').read(),  # 可以把 README.md 作为长描述
    long_description_content_type='text/markdown',  # 设置长描述的格式为 markdown
    license='Apache-2.0',  # 项目的开源协议
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
    entry_points={
        "console_scripts": ["simple_latex_ocr=simple_latex_ocr.main:main"],
    },
)
'''
python setup.py sdist bdist_wheel

twine upload dist/*
'''
