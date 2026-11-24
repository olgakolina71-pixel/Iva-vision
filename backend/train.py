"""Train IVA Vision CRNN on a line-image dataset.
CSV format: image_path,text (header required); paths relative to CSV file.
Dataset images must be single text lines. Training is an initial research prototype.
"""
import argparse, csv, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from model import IVA_CRNN, CHAR_TO_ID, VOCAB

HEIGHT=32
class Lines(Dataset):
    def __init__(self,csv_path):
        self.csv_path=Path(csv_path)
        with self.csv_path.open(encoding="utf-8-sig",newline="") as f:
            self.rows=list(csv.DictReader(f))
        self.rows=[r for r in self.rows if r.get("text") and all(c in CHAR_TO_ID for c in r["text"])]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; image=Image.open(self.csv_path.parent/row["image_path"]).convert("L")
        w=max(8,int(image.width*HEIGHT/image.height))
        image=image.resize((w,HEIGHT),Image.Resampling.BILINEAR)
        image=ImageOps.expand(image,border=1,fill=255)
        x=torch.from_numpy(__import__("numpy").array(image,dtype="float32")/255.0)
        x=(1.0-x).unsqueeze(0)
        y=torch.tensor([CHAR_TO_ID[c] for c in row["text"]],dtype=torch.long)
        return x,y,len(row["text"]),x.shape[-1]
def collate(batch):
    xs,ys,yl,xl=zip(*batch)
    maxw=max(x.shape[-1] for x in xs)
    padded=torch.zeros(len(xs),1,34,maxw)
    for i,x in enumerate(xs): padded[i,:,:,:x.shape[-1]]=x
    return padded,torch.cat(ys),torch.tensor(yl),torch.tensor(xl)
def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",required=True);p.add_argument("--epochs",type=int,default=10);p.add_argument("--batch",type=int,default=8);p.add_argument("--out",default="weights/iva_crnn.pt");args=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    data=Lines(args.csv); loader=DataLoader(data,batch_size=args.batch,shuffle=True,collate_fn=collate)
    model=IVA_CRNN().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    criterion=nn.CTCLoss(blank=0,zero_infinity=True)
    for epoch in range(args.epochs):
        model.train();total=0
        for images,targets,target_lens,widths in loader:
            images=images.to(device);targets=targets.to(device)
            logits=model(images);T=logits.shape[0]
            # CNN width reductions: two 2x pools, conv kernel at end.
            input_lens=torch.clamp((widths.to(device)//4)-1,min=1,max=T)
            loss=criterion(logits,targets,input_lens,target_lens.to(device))
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}: loss={total/max(1,len(loader)):.4f}")
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.cpu().state_dict(),"vocab":VOCAB,"height":HEIGHT},out)
    print("Saved:",out)
if __name__=="__main__": main()
