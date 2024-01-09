from app import app, clipmodel, preprocess, tokenizer, es, device
from flask import render_template, redirect, url_for, request, send_file
from app.searchForm import SearchForm
from app.inputFileForm import InputFileForm
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
# import elasticsearch
import opensearchpy
import os
from PIL import Image
import torch


INFER_ENDPOINT = "/_ml/trained_models/{model}/deployment/_infer"
INFER_MODEL_IM_SEARCH = 'sentence-transformers__clip-vit-b-32-multilingual-v1'

INDEX_IM_EMBED = 'embeddings-openclip-b-32'

HOST = app.config['ELASTICSEARCH_HOST']
AUTH = (app.config['ELASTICSEARCH_USER'], app.config['ELASTICSEARCH_PASSWORD'])
HEADERS = {'Content-Type': 'application/json'}
TLS_VERIFY = app.config['VERIFY_TLS']

app_models = {}
app_models[INFER_MODEL_IM_SEARCH] = 'started'



@app.route('/')
@app.route('/index')
def index():
    form = SearchForm()
    return render_template('index.html', title='Home', form=form)


@app.route('/image_search', methods=['GET', 'POST'])
def image_search():
    global app_models
    #is_model_up_and_running(INFER_MODEL_IM_SEARCH)

    index_name = INDEX_IM_EMBED
    if not es.indices.exists(index=index_name):
        return render_template('image_search.html', title='Image search', model_up=False,
                               index_name=index_name, missing_index=True)

    if app_models.get(INFER_MODEL_IM_SEARCH) == 'started':
        form = SearchForm()

        # Check for  method
        if request.method == 'POST':

            if 'find_similar_image' in request.form and request.form['find_similar_image'] is not None:
                image_id_to_search_for = request.form['find_similar_image']
                form.searchbox.data = None

                # image_info = es.search(
                #     index=INDEX_IM_EMBED,
                #     query={
                #         "term": {
                #             "image_id": {
                #                 "value": image_id_to_search_for,
                #                 "boost": 1.0
                #             }
                #         }
                #     },
                #     source=True)

                image_info = es.search(
                    index=INDEX_IM_EMBED,             
                    body={ 
                        "query":{
							"term": {
								"cover_id": {
									"value": image_id_to_search_for,
									"boost": 1.0
								}
							}
                    	}
                    },
                    _source=True)

                if (image_info is not None):

                    found_image = image_info['hits']['hits'][0]["_source"]
                    found_image_embedding = found_image['cover_embeddings']
                    search_response = knn_search_images(found_image_embedding)

                    return render_template('image_search.html', title='Image Search', form=form,
                                           found_image = found_image,
                                           search_results=search_response['hits']['hits'],
                                           query=form.searchbox.data, model_up=True,
                                           image_id_to_search_for=image_id_to_search_for)

            if form.validate_on_submit():
                #embeddings = sentence_embedding(form.searchbox.data)
                # embeddings = sentence_embedding_ex(form.searchbox.data, img_model)
                # embeddings = sentence_embedding_ml(form.searchbox.data, ml_model, ml_tokenizer)
                embeddings = clip_text_embedding(form.searchbox.data, clipmodel, tokenizer, device).tolist()
                print(embeddings)
                search_response = knn_search_text(form.searchbox.data, embeddings)

                return render_template('image_search.html', title='Image search', form=form,
                                       search_results=search_response['hits']['hits'],
                                       query=form.searchbox.data,  model_up=True)

            else:
                return redirect(url_for('image_search'))
        else:  # GET
            return render_template('image_search.html', title='Image search', form=form, model_up=True)
    else:
        return render_template('image_search.html', title='Image search', model_up=False, model_name=INFER_MODEL_IM_SEARCH)


@app.route('/reverse_search', methods=['GET', 'POST'])
def reverse_search():
    index_name = INDEX_IM_EMBED
    
    if not es.indices.exists(index=index_name):
        return render_template('reverse_search.html', title='Reverse Image Search', index_name=index_name, missing_index=True)

    #is_model_up_and_running(INFER_MODEL_IM_SEARCH)

    if app_models.get(INFER_MODEL_IM_SEARCH) == 'started':
        form = InputFileForm()
        if request.method == 'POST':
            if form.validate_on_submit():
                if request.files['file'].filename == '':
                    return render_template('reverse_search.html', title='Reverse Image Search', form=form,
                                           err='No file selected', model_up=True)

                filename = secure_filename(form.file.data.filename)

                url_dir = 'static/tmp-uploads/'
                upload_dir = 'app/' + url_dir
                upload_dir_exists = os.path.exists(upload_dir)
                if not upload_dir_exists:
                    # Create a new directory because it does not exist
                    os.makedirs(upload_dir)

                # physical file-dir path
                file_path = upload_dir + filename
                # relative file path for URL
                url_path_file = url_dir + filename
                # Save the image
                form.file.data.save(upload_dir + filename)

                image = Image.open(file_path)
                embedding = clip_image_embedding(image, preprocess, clipmodel, device)

                # Execute KN search over the image dataset
                search_response = knn_search_images(embedding.tolist())

                # Cleanup uploaded file after not needed
                # if os.path.exists(file_path):
                #     os.remove(file_path)

                return render_template('reverse_search.html', title='Reverse Image Search', form=form,
                                       search_results=search_response['hits']['hits'],
                                       original_file=url_path_file, filename=filename, model_up=True)
            else:
                return redirect(url_for('reverse_search'))
        else:
            return render_template('reverse_search.html', title='Reverse Image Search', form=form, model_up=True)
    else:
        return render_template('reverse_search.html', title='Reverse Image Search', model_up=False,
                               model_name=INFER_MODEL_IM_SEARCH)


@app.route('/image/<path:image_name>')
def get_image(image_name):
    try:
        # Use os.path.join to handle subdirectories
        image_path = os.path.join('./static/images/', image_name)
        return send_file(image_path, mimetype='image/jpg')
    except FileNotFoundError:
        return 'Image not found.'


@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def app_handle_413(e):
    return render_template('error.413.html', title=e.name, e_name=e.name, e_desc=e.description,
                           max_bytes=app.config["MAX_CONTENT_LENGTH"])


def sentence_embedding(query: str):
    response = es.ml.infer_trained_model(model_id=INFER_MODEL_IM_SEARCH, docs=[{"text_field": query}])
    return response['inference_results'][0]

def sentence_embedding_ex(query: str, model):
    text_emb = {}
    text_emb['predicted_value'] = model.encode(query)
    return text_emb

def sentence_embedding_ml(query: str, model, tokenizer):
	embeddings = model.forward(query, tokenizer).detach().numpy()

	text_emb = {}
	text_emb['predicted_value'] = embeddings[0]
	return text_emb
 


def knn_search_images(dense_vector: list):
    source_fields = ["product_id", "product_name", "product_shortdescription", "manufacturer_name", "cover_id", "cover_name"]
    # query = {
    #     "field": "image_embeddings",
    #     "query_vector": dense_vector,
    #     "k": 60,
    #     "num_candidates": 120,
    #     # "filter": {
	# 	# 	"multi_match" : {
	# 	# 		"query":    text, 
    #     #         "type": "cross_fields",
	# 	# 		"fields": [ "manufacturer_name", "product_name", "product_shortdescription" ],
    #     #         "minimum_should_match": "1",
    #     #         "operator":   "or"
	# 	# 	}
	# 	# }
	   
    # }
    # response = es.search(
    #     index=INDEX_IM_EMBED,
    #     fields=source_fields,
    #     knn=query,
	#     source=False)
    
    body = {
        "size":50,
        "fields": source_fields,
        "query":{
            "bool":{
				"should" : [
                    {
					"knn": {
						"cover_embeddings":{
							"k": 60,
							"vector": dense_vector
						}
					}}
				] 
			}
        }
	}

    response = es.search(
        index=INDEX_IM_EMBED,
        body=body,
        _source=False)
    print(f"Took: {response['took']} ms, Total results: {response['hits']['total']['value']}")
    return response

def knn_search_text(text, dense_vector: list):
    source_fields = ["product_id", "product_name", "product_shortdescription", "manufacturer_name", "cover_id", "cover_name"]

    body = {
        "size":200,
        "fields": source_fields,
        "query":{
            "bool":{
				"must" : [
                    {
					"knn": {
						"cover_embeddings":{
							"k": 50,
							"vector": dense_vector
						}
					}}
                    # ,
                    # {
					# "knn": {
					# 	"text_embeddings":{
					# 		"k": 60,
					# 		"vector": dense_vector
					# 	}
					# }}
                    # ,
                 	# {
                    # "multi_match" : {
					# 	"query": text, 
					# 	"type": "cross_fields",
					# 	"fields": [ "manufacturer_name", "product_name", "product_shortdescription" ],
					# 	"operator":   "or"
					# }}
				] 
			}
        }
	}

    response = es.search(
        index=INDEX_IM_EMBED,
        body=body,
        _source=False)
    print(body)
    print(f"Took: {response['took']} ms, Total results: {response['hits']['total']['value']}")
    return response


def infer_trained_model(query: str, model: str):
    response = es.ml.infer_trained_model(model_id=model, docs=[{"text_field": query}])
    return response['inference_results'][0]


def image_embedding(image, model):
    return model.encode(image)


def is_model_up_and_running(model: str):
    global app_models

    try:
        rsp = es.ml.get_trained_models_stats(model_id=model)
        if "deployment_stats" in rsp['trained_model_stats'][0]:
            app_models[model] = rsp['trained_model_stats'][0]['deployment_stats']['state']
        else:
            app_models[model] = 'down'
    except opensearchpy.NotFoundError:
        app_models[model] = 'na'

def sentencetransformer_image_embedding(image, model):
    return model.encode(image)

def clip_image_embedding(image, preprocess, model, device):
    pre = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(pre)[0]
    return image_features

def transformer_image_embedding(image, model, processor, device):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs.to(device))[0]
    return image_features

def clip_text_embedding(text, model, tokenizer, device):
    tokens = tokenizer(text).to(device)
    return model.encode_text(tokens)[0]