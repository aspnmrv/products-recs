## Запуск:

```bash
sh run_docker.sh
```

## Пример запроса

Построение модели

```bash
curl --request POST \
  --url http://localhost:4444/update_model \
  --header 'content-type: application/json' \
  --data '{"params": {"K": 10,"top_N": 10,"min_support": 0.005,"metric": "lift","min_threshold": 2}}'
```

Запрос рекоммендаций

```bash
curl --request POST \
  --url http://localhost:4444/recommend \
  --header 'content-type: application/json' \
  --data '{"bucket": [6179,6182],"id_user": 0,"params": {"n": 10,"rec_by_algo_proportion": 0.6,"return_names": 0}}'
```