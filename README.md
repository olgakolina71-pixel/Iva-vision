# IVA Vision
**Offline browser prototype with an experimental custom neural character classifier.** No Tesseract, EasyOCR, PaddleOCR, OCR APIs, or third-party OCR engines.

## Current browser demo
Open the GitHub Pages site, press **Train model in browser**, wait for synthetic character training, then upload a simple printed-text image and press Recognize. The model is created and trained at runtime using a small MLP on synthetic glyphs drawn with browser canvas. No API/backend is required.

## Limitations
This is an educational proof-of-concept, not accurate OCR yet. It segments characters by dark pixel columns and recognizes individual glyphs. It may misread characters, spacing, fonts, and complex layouts. It is not trained on real scanned/photo data and does not detect text regions robustly. Training is temporary (restarts if page reloads). For real accuracy, we need to collect/create a dataset, train a stronger model, and evaluate it.

## Optional Python research backend
The `backend/` folder contains a separate CRNN training/API scaffold. It also requires training data and weights; it is not needed by the browser demo.

## GitHub Pages
Settings → Pages → Build and deployment → Source: GitHub Actions.
