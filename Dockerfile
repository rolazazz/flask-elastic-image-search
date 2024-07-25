# docker build -t vsearch:beta4 .
# docker run -d -p 5001:5001 --network visualsearch-net -v model-cache:/root/.cache --env ES_HOST="https://opensearch-node1:9200" --env APPLICATION_ROOT="/lens" --name vsearch4 robertolazazzera/vsearch:beta4

# push to registry (docker login -u robertolazazzera)
# docker tag vsearch:beta4 robertolazazzera/vsearch:beta4
# docker push robertolazazzera/vsearch:beta4



#FROM python:3.9.13

#RUN apt-get update && rm -rf /var/lib/apt/lists/*


FROM bitnami/pytorch:latest
USER root 


WORKDIR /nlp-app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt 
	#&& \
    # Get the models from Hugging Face to bake into the container
    #python download_models.py

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5001" ]