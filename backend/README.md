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

## Train from real datasets
### CSV dataset
Prepare a CSV with `image_path,text` rows and single-line images:
```bash
python train.py --csv dataset/train.csv --epochs 10 --batch 8
```

### Hugging Face streaming dataset
The trainer can stream image/text samples without downloading the entire repository:
```bash
pip install -r requirements.txt
python train.py --hf-dataset Timka28/cyrillic --hf-split train --max-samples 2000 --epochs 2 --batch 4
```
Start with a small sample to test the pipeline. For a larger run, raise `--max-samples` to 10000 or set it to 0 for unlimited streaming. The dataset is mixed Cyrillic handwriting; incompatible characters are skipped and text is lowercased. Streaming still downloads image data as it is read, and training is compute-intensive, especially on CPU. The checkpoint is saved to `weights/iva_crnn.pt`.

Dataset reference: [Timka28/cyrillic on Hugging Face](https://huggingface.co/datasets/Timka28/cyrillic). It aggregates several Cyrillic handwriting sources. The separate CoMNIST dataset contains isolated letter images and is useful for character-classifier training, but not directly for this line-level CRNN.

## Current limits
- Prototype recognizes a single cropped text line, not full page layouts.
- No text-box detector, rotation correction, language model, or dataset augmentations yet.
- Character inventory is fixed in `model.py`.
- Dataset quality and size strongly affect accuracy.
- Do not expect accurate OCR until the weights are trained and tested on a held-out validation set.
