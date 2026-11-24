"""IVA Vision: original CRNN OCR architecture (not a wrapper around OCR engines)."""
import torch
from torch import nn

# Character inventory: Russian + English + digits + common punctuation.
CHARS = " абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()[]{}+-=*/_@#%&'\\""
BLANK = 0
VOCAB = ["<blank>"] + list(dict.fromkeys(CHARS))
CHAR_TO_ID = {c:i for i,c in enumerate(VOCAB)}
ID_TO_CHAR = {i:c for c,i in CHAR_TO_ID.items()}

class IVA_CRNN(nn.Module):
    """CNN feature extractor + bidirectional LSTM + CTC logits."""
    def __init__(self, num_classes=len(VOCAB)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(512,512,2), nn.ReLU(inplace=True)
        )
        self.rnn = nn.LSTM(512,256,num_layers=2,bidirectional=True,batch_first=True,dropout=.15)
        self.head = nn.Linear(512,num_classes)
    def forward(self,x):
        # Expected image tensor [batch, 1, 32, width].
        z=self.cnn(x)                 # [B,512,1,T]
        z=z.squeeze(2).permute(0,2,1) # [B,T,512]
        z,_=self.rnn(z)
        return self.head(z).log_softmax(-1).permute(1,0,2) # [T,B,C] for CTC
