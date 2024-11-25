# Setup pgvector
docker pull pgvector/pgvector:pg16

docker volume create pgvector-data

docker run --name pgvector-container -e POSTGRES_PASSWORD=password -p 5432:5432 -v pgvector-data:/var/lib/postgresql/data -d pgvector/pgvector:pg16

docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' pgvector-container



# Setup pgAdmin
docker pull dpage/pgadmin4

docker run --name pgadmin-container -p 5050:80 -e PGADMIN_DEFAULT_EMAIL=user@domain.com -e PGADMIN_DEFAULT_PASSWORD=password -d dpage/pgadmin4

# Setup Text Embeddings Inference (TEI)
clone this repo: https://github.com/huggingface/text-embeddings-inference

open bash \ 
model=BAAI/bge-large-en-v1.5 \
volume=$PWD/data

# TEI with GPU support
docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model

# TEI with CPU support
 docker run -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 --model-id $model
