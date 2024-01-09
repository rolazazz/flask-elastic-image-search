# docker build -t vsearch:beta2 .
# docker run -d -p 5001:5001 --network visualsearch-net -v model-cache:/root/.cache --env ES_HOST="https://opensearch-node1:9200" --env APPLICATION_ROOT="/lens" --name vsearch2 robertolazazzera/vsearch:beta2

# push to registry (docker login -u robertolazazzera)
# docker tag vsearch:beta2 robertolazazzera/vsearch:beta2
# docker push robertolazazzera/vsearch:beta2



FROM python:3.9.13

RUN apt-get update && rm -rf /var/lib/apt/lists/*

WORKDIR /nlp-app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt 
	#&& \
    # Get the models from Hugging Face to bake into the container
    #python download_models.py

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5001" ]