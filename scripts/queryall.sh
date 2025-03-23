# python chat.py -p "$2" -e nomic-embed-text -d redisstack --indexname nomic | tee ./out/nomic-redis.txt
# python chat.py -p "$2" -e nomic-embed-text -d chroma --indexname nomic | tee ./out/nomic-chroma.txt
# python chat.py -p "$2" -e nomic-embed-text -d qdrant --indexname nomic | tee ./out/nomic-qdrant.txt

# python chat.py -p "$1" -e all-minilm -d redisstack --indexname minilm | tee ./out/minilm-redis.txt
# python chat.py -p "$1" -e all-minilm -d chroma --indexname minilm | tee ./out/minilm-chroma.txt
# python chat.py -p "$1" -e all-minilm -d qdrant --indexname minilm | tee ./out/minilm-qdrant.txt
#
echo "====REDIS===="
python chat.py -p "$1" -e mxbai-embed-large -d redisstack --indexname bread
echo "====CHROMA===="
python chat.py -p "$1" -e mxbai-embed-large -d chroma --indexname bread 
echo "====QDRANT===="
python chat.py -p "$1" -e mxbai-embed-large -d qdrant --indexname bread
