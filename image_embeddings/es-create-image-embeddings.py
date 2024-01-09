import os
import sys
import glob
import time
import json
import argparse
# from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from elasticsearch import Elasticsearch, SSLError
from elasticsearch.helpers import parallel_bulk
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import torch
# import clip
import open_clip


ES_HOST = "https://127.0.0.1:9200/"
ES_USER = "elastic"
ES_PASSWORD = "password"
ES_TIMEOUT = 3600

DEST_INDEX = "embeddings-openclip-b-32"
DELETE_EXISTING = True
CHUNK_SIZE = 100

PATH_TO_IMAGES = "C:\\AppData\\product-images\\"
PREFIX = "app\\static\\product-images\\"

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
parser.add_argument('--chunk_size', dest='chunk_size', required=False, default=CHUNK_SIZE, type=int,
                    help="Default: " + str(CHUNK_SIZE))
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


def main():
    global args
    lst = []

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
    # clipmodel, preprocess = clip.load("ViT-L/14", device=device)
    # clipmodel, preprocess = clip.load("C:\\Repos\\finetuning\\model_checkpoint\\model_10.pt", device=device)
    
	# OpenClip
    clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    # clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
    # clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='C:\\Repos\\finetuning\\logs\\ciccio\\checkpoints\\epoch_3.pt', device=device)
    # clipmodel, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32', pretrained='laion5b_s13b_b90k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')


    duration = time.perf_counter() - start_time
    print(f'Duration load model = {duration}')
    
    with open('.\image_embeddings\catalog-product-most-recent-25k.json',  encoding="utf8") as user_file:
        parsed_json = json.load(user_file)[:25000]
        
    #for row in tqdm(parsed_json, desc='Processing json', total=len(parsed_json)):
    for row in tqdm(parsed_json, desc='Processing json', total=len(parsed_json)):
        try:

            def image_map(data):
                filename = f"b_{data['FileName']}"
                filepath = f"{PATH_TO_IMAGES}{filename}"
                if os.path.exists(filepath):
                    image = Image.open(filepath)
                    embeddings = clip_image_embedding(image, preprocess, clipmodel, device).tolist()
                    return {
                        'image_id':filename,
                        'image_filename': os.path.basename(filename),
                        # 'relative_path': os.path.relpath(filepath),
                        'image_embedding': embeddings
                    }
                
            filename = f"b_{row['CoverImage']['FileName']}"
            filepath = f"{PATH_TO_IMAGES}{filename}"
            image = Image.open(filepath)
            embeddings = clip_image_embedding(image, preprocess, clipmodel, device).tolist()
            text = f"{row['Name']['Value']['en']} {row['ShortDescription']['Value']['en']} by {row['Manufacturer']['Name']}"
            doc = {
                'product_id' : row['_id'],
                'product_name': row['Name']['Value']['en'],
                'product_shortdescription': row['ShortDescription']['Value']['en'],
                'manufacturer_name': row['Manufacturer']['Name'],
                'cover_id': filename,
                'cover_name': os.path.basename(filename),
                'cover_embeddings': embeddings,
                # 'relative_path': os.path.relpath(filepath).split(PREFIX)[1],
                'images': [image_map(x) for x in row['Images']],
                'text_embeddings': clip_text_embedding(text, clipmodel, tokenizer, device).tolist()
            }
            
            lst.append(doc)
            
        except Exception as e:
            print ("Unexpected Error")
            raise 
    

    duration = time.perf_counter() - start_time
    print(f'Duration creating image embeddings = {duration}')

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

    # index name to index data into
    index = DEST_INDEX
    try:
        with open(".\image_embeddings\es-index-mappings.json", "r") as config_file:
            config = json.loads(config_file.read())
            if args.delete_existing:
                if es.indices.exists(index=index):
                    print("Deleting existing %s" % index)
                    es.indices.delete(index=index, ignore=[400, 404])

            print("Creating index %s" % index)
            es.indices.create(index=index,
                              mappings=config["mappings"],
                              settings=config["settings"],
                              ignore=[400, 404],
                              request_timeout=args.timeout)


        count = 0
        for success, info in parallel_bulk(
                client=es,
                actions=lst,
                thread_count=4,
                chunk_size=args.chunk_size,
                timeout='%ss' % 120,
                index=index
        ):
            if success:
                count += 1
                if count % args.chunk_size == 0:
                    print('Indexed %s documents' % str(count), flush=True)
                    sys.stdout.flush()
            else:
                print('Doc failed', info)

        print('Indexed %s documents' % str(count), flush=True)
        duration = time.perf_counter() - start_time
        print(f'Total duration = {duration}')
        print("Done!\n")
    except SSLError as e:
        if "SSL: CERTIFICATE_VERIFY_FAILED" in e.message:
            print("\nCERTIFICATE_VERIFY_FAILED exception. Please check the CA path configuration for the script.\n")
            raise
        else:
            raise


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

def clip_text_embedding(text, model:CLIPModel, tokenizer, device):
    tokens = tokenizer(text).to(device)
    return model.encode_text(tokens)[0]

def create_image_id(filename):
    # print("Image filename: ", filename)
    return os.path.splitext(os.path.basename(filename))[0]

# def get_exif_date(filename):
#     with open(filename, 'rb') as f:
#         image = exifImage(f)
#         taken = f"{image.datetime_original}"
#         date_object = datetime.strptime(taken, "%Y:%m:%d %H:%M:%S")
#         prettyDate = date_object.isoformat()
#         return prettyDate

# def get_exif_location(filename):
#     with open(filename, 'rb') as f:
#         image = exifImage(f)
#         exif = {} 
#         lat = dms_coordinates_to_dd_coordinates(image.gps_latitude, image.gps_latitude_ref)
#         lon = dms_coordinates_to_dd_coordinates(image.gps_longitude, image.gps_longitude_ref)
#         return [lon, lat]


def dms_coordinates_to_dd_coordinates(coordinates, coordinates_ref):
    decimal_degrees = coordinates[0] + \
                      coordinates[1] / 60 + \
                      coordinates[2] / 3600
    
    if coordinates_ref == "S" or coordinates_ref == "W":
        decimal_degrees = -decimal_degrees
    
    return decimal_degrees



if __name__ == '__main__':
    main()
