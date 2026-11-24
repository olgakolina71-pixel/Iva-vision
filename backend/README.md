# IVA Vision — own OCR model
This backend is a new trainable CRNN (CNN + BiLSTM + CTC) architecture. It does **not** use Tesseract, EasyOCR, PaddleOCR, cloud OCR, or pretrained OCR engines. The checkpoint is not pretrained: recognition quality will be poor until training.

## Run API
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
The API returns 503 until `weights/iva_crnn.pt` exists.

## Train
Prepare a CSV with `image_path,text` rows and single-line images, then:
```bash
python train.py --csv dataset/train.csv --epochs 20 --batch 8
```
This writes `weights/iva_crnn.pt`. Training can be slow on CPU. Start with a small dataset to verify the pipeline, then expand and validate separately.

## Current limits
- Prototype recognizes a single cropped text line, not full page layouts.
- No text-box detector, rotation correction, language model, or dataset augmentations yet.
- Character inventory is fixed in `model.py`.
- Dataset quality and size strongly affect accuracy.
- Do not expect accurate OCR until the weights are trained and tested on a held-out validation set.
