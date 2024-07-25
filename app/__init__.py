from flask import Flask
from config import Config
from app.prefixMiddleware import PrefixMiddleware
from torch import torch
import open_clip
from sentence_transformers import SentenceTransformer
# from transformers import CLIPProcessor, CLIPModel
from opensearchpy import OpenSearch, SSLError
from opensearchpy.helpers import parallel_bulk
# from elasticsearch import Elasticsearch


app = Flask(__name__)
app.config.from_object(Config)
app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix=app.config['APPLICATION_ROOT'])

print(f'ElasticSearch Host = {Config.ELASTICSEARCH_HOST}')

# Load model, run against the image and create image embedding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Huggingface SentenceTransformer
#st_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2', device=device)
# st_model = SentenceTransformer('intfloat/e5-large-v2', device=device)
st_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)

# Huggingface Transformer
# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
# processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# OpenAI Clip
# clipmodel, preprocess = clip.load("ViT-L/14", device=device)
# clipmodel, preprocess = clip.load("C:\\Repos\\finetuning\\model_checkpoint\\model_10.pt", device=device)

# OpenClip
# clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
# clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
# clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='C:\\Repos\\finetuning\\logs\\ciccio\\checkpoints\\epoch_3.pt', device=device)
clipmodel, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32', pretrained='laion5b_s13b_b90k', device=device)
# tokenizer = open_clip.get_tokenizer('ViT-B-32')
tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')



# Load Multilanguage Model & Tokenizer
# model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
# ml_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name, resume_download=True)
# ml_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, resume_download=True)

# es = Elasticsearch(hosts=app.config['ELASTICSEARCH_HOST'],
#                    basic_auth=(app.config['ELASTICSEARCH_USER'], app.config['ELASTICSEARCH_PASSWORD']),
#                    verify_certs= app.config['VERIFY_TLS'],
#                    ca_certs='app/conf/ca.crt')
es = OpenSearch(hosts=app.config['ELASTICSEARCH_HOST'],
                http_auth=(app.config['ELASTICSEARCH_USER'], app.config['ELASTICSEARCH_PASSWORD']),
                verify_certs= app.config['VERIFY_TLS'],
				#    ,ca_certs='app/conf/ca.crt'
				use_ssl = True,
            	ssl_assert_hostname = False,
    			ssl_show_warn = False)



from app import routes

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=5001)
    
