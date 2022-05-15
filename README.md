## Запуск:

```bash
docker build -t product-recs .
docker run -d --restart unless-stopped \
           -p 5000:5000 \
           -v $(pwd)/config.json:/app/config.json \
           --name product-recs product-recs
```

## Пример запроса

Построение модели

```bash
curl --request POST \
  --url http://localhost:5000/update_model \
  --header 'content-type: application/json' \
  --data '{"id_region": 2,"params": {"K": 10,"top_N": 10,"min_support": 0.005,"metric": "lift","min_threshold": 2}}'
```

Запрос рекоммендаций

```bash
curl --request POST \
  --url http://localhost:5000/recommend \
  --header 'content-type: application/json' \
  --data '{"id_region": 2,"id_storepoint": 10,"bucket": [6179,6182],"id_user": 0,"params": {"n": 10,"rec_by_algo_proportion": 0.6,"return_names": 0}}'
```