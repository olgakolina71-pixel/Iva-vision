# IVA Vision

A lightweight browser OCR app for Russian, English and digits.

## Live app
After enabling GitHub Pages, it will be available at:
https://olgakolina71-pixel.github.io/Iva-vision/

## Features
- Image upload, drag & drop, and clipboard paste
- Russian + English OCR, with language toggles
- Numbers and punctuation included
- Editable results, copy and TXT download
- Responsive mobile-first UI
- Processing runs in browser; image isn't uploaded to an app backend

## Run
Open `index.html` via any static web server or host on GitHub Pages. OCR library and language files are downloaded from CDN on first use, so internet is needed initially.

## Engine
Prototype uses [Tesseract.js](https://github.com/naptha/tesseract.js) as an OCR engine. IVA's own trainable recognition model is planned for later iterations.

## Deploy
Repository Settings → Pages → Build and deployment → Source: GitHub Actions.
The included workflow deploys the static site when changes are pushed to `main`.

License: Apache-2.0 (project code). Tesseract.js and language model licenses apply separately.
