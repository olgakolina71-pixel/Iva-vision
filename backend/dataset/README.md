# Dataset layout
Put line-cropped training images in this folder and create `train.csv`:
```csv
image_path,text
images/line_0001.png,Пример текста 123
images/line_0002.png,Hello world 456
```
Each image should contain one text line. Supported symbols are listed in `backend/model.py`.
Use legally obtained datasets and respect their licenses. This initial pipeline does not detect text boxes or recognize whole-page layouts yet.
