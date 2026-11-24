"""Train IVA Vision CRNN from CSV or streaming Hugging Face OCR dataset.
CSV: image_path,text (single-line images, paths relative to CSV).
HF example: Timka28/cyrillic (image,text fields); streaming avoids full download.
"""
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from PIL import Image, ImageOps
from model import IVA_CRNN, CHAR_TO_ID, VOCAB

HEIGHT=32

def prep(image, text):
    image=ImageOps.grayscale(image)
    w=max(8,int(image.width*HEIGHT/image.height))
    image=image.resize((w,HEIGHT),Image.Resampling.BILINEAR)
    image=ImageOps.expand(image,border=1,fill=255)
    x=torch.from_numpy(np.array(image,dtype="float32")/255.0)
    x=(1.0-x).unsqueeze(0)
    y=torch.tensor([CHAR_TO_ID[c] for c in text],dtype=torch.long)
    return x,y,len(text),x.shape[-1]

class CsvLines(Dataset):
    def __init__(self,csv_path):
        self.csv_path=Path(csv_path)
        with self.csv_path.open(encoding="utf-8-sig",newline="") as f:
            self.rows=list(csv.DictReader(f))
        self.rows=[r for r in self.rows if r.get("text") and all(c in CHAR_TO_ID for c in r["text"])]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.csv_path.parent/r["image_path"]) as p:
            image=p.convert("L")
        return prep(image,r["text"])

class HfLines(IterableDataset):
    def __init__(self,name,split,max_samples,streaming=True,buffer_size=1000):
        from datasets import load_dataset
        self.max_samples=max_samples
        self.ds=load_dataset(name,split=split,streaming=streaming)
        if streaming:
            self.ds=self.ds.shuffle(seed=42,buffer_size=buffer_size)
    def __iter__(self):
        count=0
        for row in self.ds:
            text=str(row.get("text","")).strip().lower()
            image=row.get("image")
            if not text or image is None or len(text)>250 or not all(c in CHAR_TO_ID for c in text):
                continue
            try:
                if not isinstance(image,Image.Image): image=Image.open(image)
                yield prep(image.convert("L"),text)
                count+=1
                if self.max_samples and count>=self.max_samples: break
            except Exception:
                continue

def collate(batch):
    xs,ys,yl,xl=zip(*batch)
    maxw=max(x.shape[-1] for x in xs)
    padded=torch.zeros(len(xs),1,34,maxw)
    for i,x in enumerate(xs): padded[i,:,:,:x.shape[-1]]=x
    return padded,torch.cat(ys),torch.tensor(yl),torch.tensor(xl)

def main():
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv")
    source.add_argument("--hf-dataset",help="HF dataset ID, e.g. Timka28/cyrillic")
    p.add_argument("--hf-split",default="train")
    p.add_argument("--max-samples",type=int,default=10000,help="Streaming cap per epoch; 0=unlimited")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--out",default="weights/iva_crnn.pt")
    args=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    data=CsvLines(args.csv) if args.csv else HfLines(args.hf_dataset,args.hf_split,args.max_samples)
    loader=DataLoader(data,batch_size=args.batch,shuffle=bool(args.csv),collate_fn=collate)
    model=IVA_CRNN().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    criterion=torch.nn.CTCLoss(blank=0,zero_infinity=True)
    for epoch in range(args.epochs):
        model.train();total=0;steps=0
        for images,targets,target_lens,widths in loader:
            images=images.to(device);targets=targets.to(device)
            logits=model(images);T=logits.shape[0]
            input_lens=torch.clamp((widths.to(device)//4)-1,min=1,max=T)
            loss=criterion(logits,targets,input_lens,target_lens.to(device))
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            total+=loss.item();steps+=1
        print(f"Epoch {epoch+1}/{args.epochs}: loss={total/max(1,steps):.4f}, batches={steps}")
        if steps==0: raise RuntimeError("No compatible rows found. Check dataset fields/characters.")
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.cpu().state_dict(),"vocab":VOCAB,"height":HEIGHT},out)
    print("Saved:",out)
if __name__=="__main__": main()
