import os
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
# import Request
import pickle
import logging

from config import REGION_IDS, MAX_ITEMSET_LENGTH, AB_GROUPS
from flask import Flask, jsonify
from flask import request, send_from_directory
from Request import timeit
# from psycopg2.extras import Json
from collections import defaultdict
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, fpgrowth
from implicit.nearest_neighbours import CosineRecommender
from itertools import chain, combinations

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

COMPUTED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'computed')

if not os.path.exists(COMPUTED_PATH):
    os.mkdir(COMPUTED_PATH)
logging.basicConfig(filename=os.path.join(COMPUTED_PATH, 'log' + '.txt'), level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
FPGROWTH_METRIC_BOUNDS = {'confidence': [0, 1], 'lift': [0, np.inf], 'leverage': [-1, 1], 'conviction': [0, np.inf]}


def save_obj(obj, name: str) -> None:
    """This method saves any object in the format pickle"""
    if len(obj) > 0:
        with open(os.path.join(COMPUTED_PATH, name + '.pkl'), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name: str):
    """This method loads any object in the format pickle"""
    path = os.path.join(COMPUTED_PATH, name + '.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@timeit
def preprocess_implicit(data, K, top_N):
    """The method calculates a model for recommendations to old users.
       Model - cosine distance between item vectors"""
    users_inv_mapping = dict(enumerate(data['id_user'].unique()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}
    items_inv_mapping = dict(enumerate(data['id_item'].unique()))
    items_mapping = {v: k for k, v in items_inv_mapping.items()}

    def get_coo_matrix(df,
                       user_col='id_user',
                       item_col='id_item',
                       weight_col=None,
                       users_mapping=users_mapping,
                       items_mapping=items_mapping):
        if weight_col is None:
            weights = np.ones(len(df), dtype=np.float32)
        else:
            weights = df[weight_col].astype(np.float32)

        interaction_matrix = sp.coo_matrix((
            weights,
            (
                df[user_col].map(users_mapping.get),
                df[item_col].map(items_mapping.get)
            )
        ))
        return interaction_matrix

    def generate_implicit_recs_mapper(model, train_matrix, N,
                                      user_mapping, item_inv_mapping):
        def _recs_mapper(user):
            id_user = user_mapping[user]
            recs = model.recommend(id_user,
                                   train_matrix,
                                   N=N,
                                   filter_already_liked_items=False)
            return [int(item_inv_mapping[item]) for item in recs[0]]

        return _recs_mapper

    train_mat = get_coo_matrix(data, weight_col="freq").tocsr()
    cosine_model = CosineRecommender(K)
    cosine_model.fit(train_mat.T)
    try:
        mapper = generate_implicit_recs_mapper(cosine_model, train_mat, top_N,
                                               users_mapping, items_inv_mapping)
        recs = pd.DataFrame({
            'id_user': data['id_user'].unique()
        }, dtype=int)
        recs['id_item'] = recs['id_user'].map(mapper)
        recs.set_index('id_user', inplace=True)
    except Exception as e:
        print(e)

    return recs


@timeit
def preprocess_fpgrowth(data, min_support, metric, min_threshold):
    """The method returns a dictionary with associative rules by
       the method FP Growth"""
    te = TransactionEncoder()
    orders_encoded_array = te.fit(list(data['items'])).transform(list(data['items']))
    orders_encoded = pd.DataFrame(orders_encoded_array, columns=te.columns_)
    n_orders = orders_encoded.shape[0]

    # if min_support is too low to filter items with only one occurrence, increase it
    if min_support * n_orders < 1:
        min_support = 1.5 / n_orders

    fp = fpgrowth(orders_encoded, min_support, use_colnames=True, max_len=MAX_ITEMSET_LENGTH)
    rules = association_rules(fp, metric, min_threshold)
    rules_dict = defaultdict(list)

    for _, row in rules.iterrows():
        rules_dict[tuple(sorted(row.antecedents))] \
            .append({'consequents': list(row.consequents), 'len': len(row.antecedents), 'metric': row[metric]})

    return rules_dict


def powerset(iterable, max_subset_length=MAX_ITEMSET_LENGTH):
    """powerset({1,2,3}) --> (1,2,3) (1,2) (1,3) (2,3) (1,) (2,) (3,)"""

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(min(len(s), max_subset_length), 0, -1))


def get_categories_list(bucket, df_products):
    """The method returns a list of categories from the user's cart"""

    bucket = list(set(bucket).intersection(df_products.index))
    if len(bucket) > 0:
        return list(set([df_products.loc[bucket, 'category'].sum()]))
    else:
        return []


@timeit
def recommend_acr_categories(bucket, limit, df_acr_cat, df_products,
                             available_products_set):
    """The method adds items from product categories in
       the basket to the issuance of recommendations"""

    categories = get_categories_list(bucket, df_products)
    cat_lengths = df_acr_cat.loc[categories, 'items'].apply(lambda x: len(x))
    if cat_lengths.empty:
        max_idx = 0
    else:
        max_idx = max(cat_lengths)

    idx = 0
    result = []
    while idx < max_idx and len(result) < limit:
        for category in categories:
            if category in df_acr_cat.index and idx < len(df_acr_cat.loc[category, 'items']):
                item = df_acr_cat.loc[category, 'items'][idx]
                if item not in bucket and item not in result \
                        and item in available_products_set:
                    result.append(item)
        idx += 1
    return result[:limit]


def rules_sorting_key(item):
    """The method returns a rule for sorting association rules"""

    return item['len'] * 2 + item['metric']


@timeit
def retrieve(src, result, bucket, n, available_products_set, df_tags=None):
    """An important method that limits the issuance of rule recommendations"""
    for item in src:
        if item not in result and item not in bucket \
                and item in available_products_set:
            result.append(item)
        if len(result) == n:
            break
    return result


@timeit
def except_recommend(acr_list, bucket, n, available_products_set):
    return retrieve(acr_list, [], bucket, n, available_products_set)


@timeit
def recommend_fpgrowth(bucket, n, available_products_set):
    """The method returns items from a dictionary of association rules"""
    rules_dict = load_obj("rules_dict")
    if len(bucket) > 20:
        small_bucket = bucket[:20]
    else:
        small_bucket = bucket
    candidates = [e for subset in powerset(small_bucket) for e in rules_dict[tuple(sorted(subset))]]
    candidates.sort(key=rules_sorting_key, reverse=True)
    result = []
    for candidate in candidates:
        for id in candidate["consequents"]:
            if id not in result and id not in bucket and id in available_products_set:
                result.append(id)
        if len(result) >= n:
            break
    return result[:n]


@timeit
def recommend_new_users(bucket, n, df_products, df_acr_cat, acr_list,
                        available_products_set, rec_by_algo, df_tags=None):
    """The method returns a list of recommended items for new users"""
    ratio_log = dict()
    fp_growth_items = recommend_fpgrowth(bucket=bucket, n=n, available_products_set=available_products_set)[:rec_by_algo]

    ratio_log["fp"] = len(fp_growth_items)
    temp_log = len(fp_growth_items)
    acr_cat_items = recommend_acr_categories(bucket=bucket, limit=100, df_acr_cat=df_acr_cat,
                                             df_products=df_products, available_products_set=available_products_set)

    result = retrieve(acr_cat_items, fp_growth_items, bucket, n, available_products_set, df_tags)
    ratio_log["acr_cat"] = len(result) - temp_log
    temp_log = len(result)

    if len(result) < n:
        result = retrieve(acr_list, result, bucket, n, available_products_set, df_tags)
    ratio_log["acr"] = len(result) - temp_log
    return result, ratio_log


@timeit
def recommend_old_users(id_user, bucket, recs_knn, available_products_set, n, rec_by_algo, freq_items_list, acr_list,
                        ab_group=None, df_tags=None):
    """The method returns a list of recommended items for old users"""

    ratio_log = dict()
    result = []
    result = retrieve(recs_knn.at[id_user, 'id_item'], result, bucket, rec_by_algo,
                      available_products_set)
    ratio_log["implicit"] = len(result)
    temp_log = len(result)

    if len(result) < n or rec_by_algo != 1:
        result = retrieve(freq_items_list, result, bucket, n, available_products_set)

    ratio_log["freq_items"] = len(result) - temp_log
    temp_log = len(result)

    if len(result) < n:
        result = retrieve(acr_list, result, bucket, n, available_products_set, df_tags)

    ratio_log["acr"] = len(result) - temp_log
    return result, ratio_log


@app.route('/', methods=['GET'])
def healthcheck():
    return 'healthy', 200


@app.route('/version.json')
def version():
    return send_from_directory('.', 'version.json')


# @app.route('/start_experiment', methods=['POST'])
# def start_experiment():
#     try:
#         id_experiment = int(request.json['id'])
#         name = request.json['name']
#         if 'description' in request.json:
#             description = request.json['description']
#         else:
#             description = None
#         rows_changed = Request.start_experiment(id_experiment, name, description)
#     except Exception as e:
#         return jsonify("WRONG PARAMS"), 400
#     return jsonify(f"{rows_changed} row(s) changed"), 200
#
#
# @app.route('/end_experiment', methods=['POST'])
# def end_experiment():
#     try:
#         id_experiment = int(request.json['id'])
#         rows_changed = Request.end_experiment(id_experiment)
#     except Exception as e:
#         return jsonify("WRONG PARAMS"), 400
#     return jsonify(f"{rows_changed} row(s) changed"), 200


@app.route('/update_model', methods=['POST'])
def update_model():
    if not os.path.exists(COMPUTED_PATH):
        os.mkdir(COMPUTED_PATH)
    params = request.json["params"]

    K = int(params['K'])
    top_N = int(params['top_N'])
    min_support = float(params['min_support'])
    metric = str(params['metric'])
    min_threshold = float(params['min_threshold'])

    if metric not in FPGROWTH_METRIC_BOUNDS or min_threshold < FPGROWTH_METRIC_BOUNDS[metric][0] \
            or min_threshold > FPGROWTH_METRIC_BOUNDS[metric][1]:
        return "Metric not supported or min_threshold out of range", 200

    # df_alltime = Request.get_alltime(id_region=id_region)
    # df_year = Request.get_year(id_region=id_region)
    # df_products = Request.get_products(id_region=id_region)
    # df_freq_items = Request.get_freq_items(id_region=id_region)
    # acr_list = Request.get_acr_list(id_region=id_region)
    # df_acr_cat = Request.get_acr_cat(id_region=id_region)
    # df_tags = Request.get_tags(id_region=id_region)

    df_alltime = load_obj("df_alltime")

    df_year = load_obj("df_year")
    df_products = load_obj("df_products")
    df_freq_items = load_obj("df_freq_items")
    acr_list = load_obj("acr_list")
    df_acr_cat = load_obj("df_acr_cat")
    df_tags = load_obj("df_tags")

    if len(df_alltime) == 0:
        return jsonify("NO_DATA"), 200

    df_alltime.drop_duplicates(subset=['id_user', 'id_item'], keep='first', inplace=True)

    recs_knn = preprocess_implicit(df_alltime, K, top_N)

    rules_dict = preprocess_fpgrowth(df_year, min_support, metric, min_threshold)

    save_obj(recs_knn, "implicit_model")
    save_obj(rules_dict, "rules_dict")
    save_obj(df_products, "df_products")
    save_obj(acr_list, "acr_list")
    save_obj(df_freq_items, "df_freq_items")
    save_obj(df_acr_cat, "df_acr_cat")
    save_obj(df_tags, "df_tags")

    return jsonify("OK"), 200


@app.route('/recommend', methods=['POST'])
@timeit
def recommend():
    bucket = list(request.json["bucket"])
    id_user = int(request.json["id_user"])
    test = int(request.json["test"]) if "test" in request.json else 0
    # id_experiment = Request.get_current_experiment()
    id_experiment = None
    if id_experiment is None:
        ab_group = None
    else:
        ab_group = np.random.choice(AB_GROUPS)

    params = request.json["params"]
    n = int(params["n"])
    rec_by_algo_proportion = min(1., float(params["rec_by_algo_proportion"]))
    rec_by_algo = round(rec_by_algo_proportion * n)
    return_names = int(params["return_names"])
    acr_list = load_obj("acr_list")
    df_products = load_obj("df_products")

    recs_knn = load_obj("implicit_model")
    df_tags = load_obj("df_tags")

    if acr_list is None or df_products is None or recs_knn is None:
        return jsonify([]), 200
    # available_products_set = Request.get_active_products(id_region, id_storepoint)
    # available_products_set = Request.get_active_products(id_region, id_storepoint)
    available_products_set = set(df_products.index)

    # If user is not specified or no orders found then recommend by rules
    if id_user == 0 or id_user not in recs_knn.index:
        # type_algo = 1
        df_acr_cat = load_obj("df_acr_cat")
        if df_acr_cat is None:
            return jsonify([]), 200
        result, ratio_log = recommend_new_users(bucket=bucket, n=n, df_products=df_products, df_acr_cat=df_acr_cat,
                                                acr_list=acr_list,
                                                available_products_set=available_products_set,
                                                rec_by_algo=rec_by_algo)
    # Else recommend with implicit nearest neighbors algorithm
    else:
        # type_algo = 2
        df_freq_items = load_obj("df_freq_items")
        if not isinstance(df_freq_items, pd.DataFrame):
            return jsonify([]), 200
        freq_item_list = df_freq_items.loc[id_user, 'freq_items']
        result, ratio_log = recommend_old_users(id_user=id_user, bucket=bucket, recs_knn=recs_knn,
                                                available_products_set=available_products_set, n=n,
                                                freq_items_list=freq_item_list, acr_list=acr_list,
                                                rec_by_algo=rec_by_algo)
    id_recommendation = 0  # will be replaced if test != 0
    try:
        if test == 0:
            # id_recommendation = Request.save_logs(id_user=id_user, bucket=bucket, type_algo=type_algo, id_region=id_region,
            #                   ab_group=ab_group, id_experiment=id_experiment,
            #                   id_storepoint=id_storepoint, n=n, rec_by_algo=rec_by_algo, result=result,
            #                   ratio_log=Json(ratio_log))
            id_recommendation = random.randint(0, 10000000)
    except Exception as e:
        logger.exception(e)

    if return_names == 1:
        return jsonify(bucket=[df_products.loc[id_product, 'name'] for id_product in result],
                        id=id_recommendation), 200
    else:
        return jsonify(bucket=result, id=id_recommendation), 200
