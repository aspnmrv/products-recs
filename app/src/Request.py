import psycopg2
import pandas.io.sql as sqlio
import time
import json
import os
import datetime
from typing import List
from config import MINCONN, MAXCONN
from config import PRINT_TIMINGS
from psycopg2 import pool
from psycopg2.extras import Json
#
# CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if PRINT_TIMINGS:
            print('%r  %.4f s' % (method.__name__, (te - ts)))
        return result

    return timed
#
#
# @timeit
# def connect_from_config(file):
#     with open(file, 'r') as fp:
#         config = json.load(fp)
#     return psycopg2.connect(**config)
#
#
# @timeit
# def create_pool_from_config(minconn, maxconn, file):
#     with open(file, 'r') as fp:
#         config = json.load(fp)
#     return pool.SimpleConnectionPool(minconn, maxconn, **config)
#
#
# GLOBAL_POOL = create_pool_from_config(MINCONN, MAXCONN, CONFIG_PATH)
#
#
# @timeit
# def get_alltime(id_region):
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql_alltime = '''
#         '''.format(id_region=id_region)
#         data = sqlio.read_sql_query(sql_alltime, conn)
#     finally:
#         GLOBAL_POOL.putconn(conn)
#     return data
#
#
# @timeit
# def get_year(id_region):
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql_year = """
#         """.format(id_region=id_region)
#         data = sqlio.read_sql_query(sql_year, conn)
#     finally:
#         GLOBAL_POOL.putconn(conn)
#     return data
#
#
# @timeit
# def get_products(id_region):
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql_products = """
#         """.format(id_region=id_region)
#         data = sqlio.read_sql_query(sql_products, conn, index_col='id')
#     finally:
#         GLOBAL_POOL.putconn(conn)
#     return data
#
#
# @timeit
# def get_freq_items(id_region):
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql_freq_items = """
#         """.format(id_region=id_region)
#         data = sqlio.read_sql_query(sql_freq_items, conn, index_col='id_user')
#     finally:
#         GLOBAL_POOL.putconn(conn)
#     return data
#
#
# def get_tags(id_region):
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql_tags = f""
#         data = sqlio.read_sql_query(sql_tags, conn, index_col='id')
#     finally:
#         GLOBAL_POOL.putconn(conn)
#     return data
#
#
# @timeit
# def get_acr_list(id_region):
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql_acr = """
#         """.format(id_region=id_region)
#
#         data = sqlio.read_sql_query(sql_acr, conn)
#     finally:
#         GLOBAL_POOL.putconn(conn)
#     return list(data.id)
#
#
# @timeit
# def get_acr_cat(id_region):
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql_acr_cat = """
#         """.format(id_region=id_region)
#         data = sqlio.read_sql_query(sql_acr_cat, conn, index_col='id_productcategory')
#     finally:
#         GLOBAL_POOL.putconn(conn)
#     return data
#
#
# @timeit
# def get_active_products(id_region, id_storepoint):
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql_active_products = f"""
#         """
#         data = sqlio.read_sql_query(sql_active_products, conn)
#     finally:
#         GLOBAL_POOL.putconn(conn)
#
#     if data is not None and data.loc[0, 'products']:
#         return set(data.loc[0, 'products'])
#     else:
#         return {}
#
#
#
# def nullify(param):
#     if param is None:
#         return 'NULL'
#     if isinstance(param, str):
#         return '\'' + str(param) + '\''
#     return param
#
#
# def save_logs(id_user: int, bucket: List[int], type_algo: int,
#               id_region: int, id_experiment:int, ab_group: str, id_storepoint: int,
#               n: int, rec_by_algo: float, result: List[int], ratio_log: Json):
#
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql = f"""
#             INSERT INTO recs_log_v2 (date, user_id, bucket, type_algo, id_region, id_experiment, ab_group, id_storepoint, n, rec_by_algo, result,
#             ratio_log)
#             VALUES ('{datetime.datetime.now()}', {id_user}, ARRAY{bucket}::integer[], {type_algo}, {id_region}, {nullify(id_experiment)},
#             {nullify(ab_group)},
#      {id_storepoint}, {n}, {rec_by_algo}, ARRAY{result}::integer[], {ratio_log}) RETURNING id
#         """
#         cursor = conn.cursor()
#         cursor.execute(sql)
#         id_last_record = cursor.fetchone()[0]
#         conn.commit()
#         cursor.close()
#     finally:
#         GLOBAL_POOL.putconn(conn)
#     return id_last_record
#
#
# def get_current_experiment():
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql_experiments = """
#         select id from experiments where isactive
#         """
#         data = sqlio.read_sql_query(sql_experiments, conn)
#     finally:
#         GLOBAL_POOL.putconn(conn)
#
#     if len(data) == 1:
#         return data.at[0, 'id']
#     else:
#         return None
#
#
# def start_experiment(id_experiment, name, description=None):
#     conn = GLOBAL_POOL.getconn()
#     try:
#         sql = f"""
#         INSERT INTO experiments
#         VALUES
#         ({id_experiment}, '{name}', current_timestamp, NULL, true, {nullify(description)})
#         """
#         cursor = conn.cursor()
#         cursor.execute(sql)
#         rows_changed = cursor.rowcount
#         conn.commit()
#         cursor.close()
#     finally:
#         GLOBAL_POOL.putconn(conn)
#     return rows_changed
#
#
# def end_experiment(id_experiment):
#     conn = GLOBAL_POOL.getconn()
#     rows_changed = 0
#     try:
#         sql = f"""
#         UPDATE experiments
#         SET dateend = current_timestamp,
#         isactive = false
#         WHERE id = {id_experiment}
#         """
#         cursor = conn.cursor()
#         cursor.execute(sql)
#         rows_changed = cursor.rowcount
#         conn.commit()
#         cursor.close()
#     finally:
#         GLOBAL_POOL.putconn(conn)
#         return rows_changed
