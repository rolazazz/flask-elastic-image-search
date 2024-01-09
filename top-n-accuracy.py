import os
import sys
import glob
import time
import json
import argparse
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from elasticsearch import Elasticsearch, SSLError
from elasticsearch.helpers import parallel_bulk
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch
import clip
import open_clip


ES_HOST = "https://127.0.0.1:9200/"
ES_USER = "elastic"
ES_PASSWORD = "password"
ES_TIMEOUT = 3600

INDEX_IM_EMBED = 'embeddings-openclip-xlm-base'

PATH_TO_IMAGES = ".\\app\\static\\variant-images\\**\\a_*.jp*g"
PREFIX = ".\\app\\static\\images\\"

CA_CERT='./app/conf/ca.crt'

parser = argparse.ArgumentParser()
parser.add_argument('--es_host', dest='es_host', required=False, default=ES_HOST,
                    help="Elasticsearch hostname. Must include HOST and PORT. Default: " + ES_HOST)
parser.add_argument('--es_user', dest='es_user', required=False, default=ES_USER,
                    help="Elasticsearch username. Default: " + ES_USER)
parser.add_argument('--es_password', dest='es_password', required=False, default=ES_PASSWORD,
                    help="Elasticsearch password. Default: " + ES_PASSWORD)
parser.add_argument('--verify_certs', dest='verify_certs', required=False, default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Verify certificates. Default: True")
parser.add_argument('--thread_count', dest='thread_count', required=False, default=4, type=int,
                    help="Number of indexing threads. Default: 4")
parser.add_argument('--timeout', dest='timeout', required=False, default=ES_TIMEOUT, type=int,
                    help="Request timeout in seconds. Default: " + str(ES_TIMEOUT))
parser.add_argument('--delete_existing', dest='delete_existing', required=False, default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Delete existing indices if they are present in the cluster. Default: True")
parser.add_argument('--ca_certs', dest='ca_certs', required=False, default=CA_CERT,
                    help="Path to CA certificate.") # Default: ../app/conf/ess-cloud.cer")
parser.add_argument('--extract_GPS_location', dest='gps_location', required=False, default=False,
                    action=argparse.BooleanOptionalAction,
                    help="[Experimental] Extract GPS location from photos if available. Default: False")

args = parser.parse_args()

es = Elasticsearch(hosts=ES_HOST)
if args.ca_certs:
	es = Elasticsearch(
		hosts=[args.es_host],
		verify_certs=args.verify_certs,
		basic_auth=(args.es_user, args.es_password),
		ca_certs=args.ca_certs
	)
else:
	es = Elasticsearch(
		hosts=[args.es_host],
		verify_certs=args.verify_certs,
		basic_auth=(args.es_user, args.es_password)
	)

es.options(request_timeout=args.timeout)

def main():
    global args
    lst = []
    top1 = 0
    top3 = 0
    top5 = 0
    top10= 0
    count_img = 0

    start_time = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Huggingface SentenceTransformer
    #st_model = SentenceTransformer('clip-ViT-L-14', device=device)

	# Huggingface Transformer
    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    # processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

	# OpenAI Clip
    #clipmodel, preprocess = clip.load("ViT-L/14", device=device)
    
	# OpenClip
    # clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    # clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
    clipmodel, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32', pretrained='laion5b_s13b_b90k', device=device)

    duration = time.perf_counter() - start_time
    print(f'Duration load model = {duration}')


    start_time = time.perf_counter()
    with open('.\image_embeddings\catalog-product-most-recent-25k.json',  encoding="utf8") as user_file:
        parsed_json = json.load(user_file)
        
    for row in tqdm(parsed_json[:25000], desc='Processing json', total=len(parsed_json[:25000])):
        for image in(row["Images"]):
            filename = f".\\app\\static\product-images\\b_{image['FileName']}"
            
            if image["Type"] == 0 and image["FileName"]!=row["CoverImage"]["FileName"] and Path(filename).exists():
                product_id = row['_id']
                image_handle = Image.open(filename)
                # print(os.path.relpath(filename))
                # Estract embeddings
				# embedding = sentencetransformer_image_embedding(image_handle, st_model)
                # embedding = transformer_image_embedding(image_handle, model, processor, device)
                embedding = clip_image_embedding(image_handle, preprocess, clipmodel, device)
                # Execute KN search over the image dataset
                search_response = knn_search_images(embedding.tolist())
                hits = search_response['hits']['hits']
                found = False;
                for i in range(len(hits)):
                    if hits[i]['fields']['product_id'][0] == str(product_id):
                        #   print(f"image {image['FileName']} of {product_id} at position {i}")
                          found = True
                          if i==0: top1+=1
                          if i<=2: top3+=1
                          if i<=4: top5+=1
                          if i<=9: top10+=1
                          break
                # if found == False:
                #     print(f'image {image["FileName"]} of {product_id} not found')
                    
                count_img+=1
                
                    
		



    duration = time.perf_counter() - start_time
    print(f'Duration Top-N Accuracy algorithm = {duration}, Image Count = {count_img}')
    print(f'Top1: {top1/count_img}; Top3: {top3/count_img}; Top5:{top5/count_img}; Top10: {top10/count_img}')





def sentencetransformer_image_embedding(image, model):
    return model.encode(image)

def clip_image_embedding(image, preprocess, model, device):
    pre = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(pre)[0]
    return image_features

def transformer_image_embedding(image, model:CLIPModel, processor:CLIPProcessor, device):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs.to(device))[0]
    return image_features

def create_image_id(filename):
    # print("Image filename: ", filename)
    return os.path.splitext(os.path.basename(filename))[0]

def knn_search_images(dense_vector: list):
    source_fields = ["product_id", "image_id", "image_name", "relative_path"]
    query = {
        "field": "image_embedding",
        "query_vector": dense_vector,
        "k": 10,
        "num_candidates": 10
    }

    response = es.search(
        index=INDEX_IM_EMBED,
        fields=source_fields,
        knn=query, source=False)

    return response

if __name__ == '__main__':
    main()
