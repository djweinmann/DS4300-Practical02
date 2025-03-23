# python ingest.py -e nomic-embed-text -d redisstack --indexname nomic
# python ingest.py -e nomic-embed-text -d chroma --indexname nomic
# python ingest.py -e nomic-embed-text -d qdrant --indexname nomic
#
# python ingest.py -e all-minilm -d redisstack --indexname minilm
# python ingest.py -e all-minilm -d chroma --indexname minilm
# python ingest.py -e all-minilm -d qdrant --indexname minilm

python ingest.py -e mxbai-embed-large -d redisstack --indexname bread
python ingest.py -e mxbai-embed-large -d chroma --indexname bread
python ingest.py -e mxbai-embed-large -d qdrant --indexname bread
