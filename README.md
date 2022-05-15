## Description:

```text
This project is an application that functions as a recommender system.
For users for whom we know the purchase history, the algorithm builds recommendations based on collaborative filtering (cosine distance). 
For new users, the FPGrowth association rules algorithm is used.

/update_model

This command builds a recommendation model based on the interaction matrix as well as other 
inputs such as purchase frequency and category matching.

/recommend

This command returns recommendations for a user from a file with an interaction matrix based 
on the collaborative filtering algorithm (implicit) or for a new user based on the FPGrowth associative rules algorithm.
```

## Start:

```bash
sh run_docker.sh
```

## Example Request

Build Model

```bash
curl --request POST \
  --url http://localhost:4444/update_model \
  --header 'content-type: application/json' \
  --data '{"params": {"K": 10,"top_N": 10,"min_support": 0.005,"metric": "lift","min_threshold": 2}}'
```

Recommend Request

```bash
curl --request POST \
  --url http://localhost:4444/recommend \
  --header 'content-type: application/json' \
  --data '{"bucket": [6179,6182],"id_user": 0,"params": {"n": 10,"rec_by_algo_proportion": 0.6,"return_names": 0}}'
```