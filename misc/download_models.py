from torch import torch
from sentence_transformers import SentenceTransformer
from multilingual_clip import pt_multilingual_clip
import transformers


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_model = SentenceTransformer('clip-ViT-L-14', device=device)

# Load Model & Tokenizer
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
ml_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name, device=device, resume_download=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, device=device, resume_download=True)