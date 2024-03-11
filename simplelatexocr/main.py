#!/usr/bin/env python
from simplelatexocr.models import Latex_OCR
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config_path", type=str, default=None)
    parser.add_argument("-detecter", "--detect_path", type=str, default=None)
    parser.add_argument("-encoder", "--encoder_path", type=str, default=None)
    parser.add_argument("-decoder", "--decoder_path", type=str, default=None)
    parser.add_argument("-tokenizer", "--tokenizer_json", type=str, default=None)
    parser.add_argument("img_path", type=str, help="Only img path of the formula.")
    args = parser.parse_args()

    engine = Latex_OCR()

    result = engine.predict(args.img_path)
    print(result['formula'])
    print(result['confidence'])
    print(result['elapse'])


if __name__ == "__main__":
    main()
