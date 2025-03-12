import pytesseract

# 指定 Tesseract 主程序路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 指定 tessdata 目录（解决路径混合斜杠问题）
custom_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'

# 示例调用
text = pytesseract.image_to_string('image.png', lang='eng', config=custom_config)
print(text)