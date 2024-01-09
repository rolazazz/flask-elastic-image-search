import os
import sys
import glob
import time
import json
import argparse

from PIL import Image
from tqdm import tqdm
from datetime import datetime

import open_clip
import torch
import numpy as np





DEST_INDEX = ""
DELETE_EXISTING = True
CHUNK_SIZE = 100

PATH_TO_IMAGES = "../app/static/images/**/*.jp*g"
PREFIX = "app\\static\\product-images\\"

CA_CERT='./app/conf/ca.crt'

parser = argparse.ArgumentParser()

args = parser.parse_args()


def main():
	global args
	lst = []

	start_time = time.perf_counter()
	# img_model = SentenceTransformer('clip-ViT-B-32', device='cuda'
	labels = np.array(["Wall tiles","Wallpapers","Indoor flooring","Parquets","Outdoor floor tiles","Thermal insulation polymer sheets and panels","Wooden doors","Internal doors","Glass doors","Windows","Door Handles","Decorative radiators","Benches","Litter bins","Bathtubs","Bathroom cabinets","Kitchens","Sinks","Cooker hoods","Bookcases","Sofas","Chairs","TV cabinets","Beds","Wardrobes","Office desks","Office chairs","Pendant lamps","Wall lamps","Table lamps","Floor lamps","Ceiling lamps","Decorative radiators","Rugs","Storage walls","Armchairs","Poufs","Console Tables","Footstools","Fireplaces","Stoves","Stools","Indoor benches","Tables","Coffee tables","Writing desks","Highboards","Sideboards","Chests of drawers","Bedside tables","Kitchen Taps","Hobs","Vanity units","Washbasins","Shower trays","Shower cabins","Shower panels","Overhead showers","Toilets","Bidets","Bathroom mirrors","Mirrors","Fabrics","Cushions","Coat racks","Garden sofas","Garden chairs","Garden armchairs","Garden tables","Garden side tables","Meeting tables","Executive chairs","Training chairs","Easy chairs","Visitor's chairs","Sun loungers","Waiting room chairs","Lounge chairs","Barstools","Waiting room sofas","Lounge tables","Hotel sofas","high tables","Restaurant chairs","Restaurant tables","Spotlights","Outdoor wall lamps","Bollard lights","Toilet roll holders","Soap dishes","Toothbrush holders","Toilet brushes","Towel racks","Mosaics","Mosaics","Wardrobes","Chests of drawers","Trays","Decorative objects","Vases","Paintings","Washbasin taps","Bathtub taps","Shower taps","Handshowers","Bidet taps","3D Wall Claddings","Garden stools","Window handles","Spouts taps","Door panels","Furniture foils","Curtains Fabrics","Upholstery fabrics","Acoustic wall panels","Safety shoes","Work clothes","Steplights","Outdoor steplights","Track-Lights","Towel warmers","Overhead showers","Shower panels","Lounge armchairs","Hinged doors","Acoustic wall panels","Desk lamps","Safety shoes","Work clothes","Sanitisable wallpapers"])
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model, _, transform = open_clip.create_model_and_transforms('coca_ViT-L-14', pretrained='mscoco_finetuned_laion2B-s13B-b90k', device=device)
	tokenizer = open_clip.get_tokenizer('ViT-L-14')
	duration = time.perf_counter() - start_time
	print(f'Duration load model = {duration}')
    
	text = tokenizer(labels).to(device)
	text_features = model.encode_text(text).float()
	text_features /= text_features.norm(dim=-1, keepdim=True)

	with open('.\image_embeddings\catalog-product-most-recent-25k.json',  encoding="utf8") as user_file:
		parsed_json = json.load(user_file)
		
	for row in tqdm(parsed_json[:100], desc='Processing json', total=len(parsed_json[:100])):
		path_name = f".\\app\\static\product-images\\b_{row['CoverImage']['FileName']}"
		print(path_name)
		image = transform(Image.open(path_name)).unsqueeze(0).to(device)
		# text = tokenizer(labels).to(device)

		with torch.no_grad(), torch.cuda.amp.autocast():
                        
			# https://github.com/mlfoundations/open_clip/issues/575		
			generated = model.generate(image, num_beam_groups=1)
			caption = open_clip.decode(generated[0])
			print(caption)


			# image_features = model.encode_image(image).float()
			# # text_features = model.encode_text(text).float()
			# image_features /= image_features.norm(dim=-1, keepdim=True)
			# # text_features /= text_features.norm(dim=-1, keepdim=True)

			# text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
			# top_probs, top_labels = text_probs.cpu().topk(3, dim=-1)
			# print("Label probs:", labels[top_labels[0].numpy()], top_probs.numpy()[0]) 
                        
						
                        
			# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
			# values, indices = similarity[0].topk(3)
			# print("Top predictions:", zip(values, indices))
			# for value, index in zip(values, indices):
			# 	print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")







def image_embedding(image, model):
    
    generated = model.generate(image)

    return model.encode(image)


def create_image_id(filename):
    # print("Image filename: ", filename)
    return os.path.splitext(os.path.basename(filename))[0]



if __name__ == '__main__':
    main()
