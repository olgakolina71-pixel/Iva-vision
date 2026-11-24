"""IVA Vision inference API. Requires a trained checkpoint at weights/iva_crnn.pt."""
from pathlib import Path
from io import BytesIO
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import IVA_CRNN, ID_TO_CHAR
app=FastAPI(title="IVA Vision API",version="0.1")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
device="cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path=Path("weights/iva_crnn.pt")
model=None
@app.get("/")
def health(): return {"service":"IVA Vision custom OCR","model_loaded":model is not None,"device":device}
def decode(logits):
    ids=logits.argmax(-1).tolist();out=[];prev=0
    for k in ids:
        if k!=prev and k!=0:out.append(ID_TO_CHAR.get(k,""))
        prev=k
    return "".join(out)
@app.post("/predict")
async def predict(image:UploadFile=File(...),languages:str=Form("rus,eng")):
    global model
    if not checkpoint_path.exists():
        raise HTTPException(503,"IVA model weights are missing. Train the model first.")
    if model is None:
        ckpt=torch.load(checkpoint_path,map_location=device,weights_only=False)
        model=IVA_CRNN().to(device);model.load_state_dict(ckpt["state_dict"]);model.eval()
    try:
        img=Image.open(BytesIO(await image.read())).convert("L")
        # Prototype expects one cropped line. Match training preprocessing: white background,
        # black glyphs become high values after inversion, with a one-pixel white border.
        h=32;w=max(8,int(img.width*h/img.height));img=img.resize((w,h))
        img=ImageOps.expand(img,border=1,fill=255)
        import numpy as np
        arr=np.asarray(img,dtype=np.float32)/255.0
        x=torch.from_numpy(1.0-arr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad(): pred=model(x)[0]
        return {"text":decode(pred),"model":"IVA-CRNN (custom)","note":"Prototype: single-line recognition"}
    except Exception as e: raise HTTPException(400,f"Image processing failed: {e}")
