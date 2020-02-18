#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

from argparse import ArgumentParser
from contextlib import contextmanager
from time import time

from pyspark import broadcast
from pyspark.sql import Row, SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *


LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))


def get_combined_model(df, frequency_limit = None):
    cols = ['_c%d' % i for i in CAT_COLS]
    df = (df
        .select(posexplode(array(*cols)))
        .withColumnRenamed('pos', 'column_id')
        .withColumnRenamed('col', 'data')
        .filter('data is not null')
        .groupBy('column_id', 'data')
        .count())

    if frequency_limit:
        frequency_limit = frequency_limit.split(",")
        exclude = []
        default_limit = None
        for fl in frequency_limit:
            frequency_pair = fl.split(":")
            if len(frequency_pair) == 1:
                default_limit = int(frequency_pair[0])
            elif len(frequency_pair) == 2:
                df = df.filter((col('column_id') != int(frequency_pair[0]) - CAT_COLS[0]) | (col('count') >= int(frequency_pair[1])))
                exclude.append(int(frequency_pair[0]))
        if default_limit:
            remain = [x - CAT_COLS[0] for x in CAT_COLS if x not in exclude]
            df = df.filter((~col('column_id').isin(remain)) | (col('count') >= default_limit))
            # for comparing isin and separate filter
            # for i in remain:
            #     df = df.filter((col('column_id') != i - CAT_COLS[0]) | (col('count') >= default_limit))

    windowed = Window.partitionBy('column_id').orderBy(desc('count'))
    return (df
        .withColumn('id', row_number().over(windowed))
        .drop('count'))


def get_column_models(combined_model):
    for i in CAT_COLS:
        model = (combined_model
            .filter('column_id == %d' % (i - CAT_COLS[0]))
            .drop('column_id'))
        yield i, model


def apply_models(df, models, broadcast_model = False):
    for i, model in models:
        col_name = '_c%d' % i
        model = broadcast(model) if broadcast_model else model
        df = (df
            .join(model, col(col_name) == col('data'), how='left')
            .drop(col_name, 'data')
            .withColumnRenamed('id', col_name))
    return df.fillna(0, ['_c%d' % i for i in CAT_COLS])


def transform_log(df, transform_log = False):
    cols = ['_c%d' % i for i in INT_COLS]
    if transform_log:
        for col_name in cols:
            df = df.withColumn(col_name, log(df[col_name] + 1))
    return df.fillna(0, cols)


def randomize(df):
    return (df
        .withColumn('random', rand())
        .orderBy('random')
        .drop('random'))


def split(df, weights):
    df = (df
        .rdd
        .zipWithIndex()
        .map(lambda x: Row(index=x[1], **x[0].asDict()))
        .toDF())
    sum = __builtins__.sum
    splits = ((sum(weights[:i]), sum(weights[:i+1])) for i in range(len(weights)))
    return (df.filter('index >= %d and index < %d' % split).drop('index') for split in splits)


def delete_data_source(spark, path):
    sc = spark.sparkContext
    config = sc._jsc.hadoopConfiguration()
    path = sc._jvm.org.apache.hadoop.fs.Path(path)
    sc._jvm.org.apache.hadoop.fs.FileSystem.get(config).delete(path, True)


def load_raw(spark, folder, days):
    label_fields = [StructField('_c%d' % LABEL_COL, IntegerType())]
    int_fields = [StructField('_c%d' % i, IntegerType()) for i in INT_COLS]
    str_fields = [StructField('_c%d' % i, StringType()) for i in CAT_COLS]

    schema = StructType(label_fields + int_fields + str_fields)

    reader = (spark
        .read
        .schema(schema)
        .option('sep', '\t')
        .csv)

    paths = [os.path.join(folder, 'day_%d' % i) for i in range(days)]
    return reader(paths), [reader(path) for path in paths]


def save_transformed(df, folder, day, mode=None):
    path = os.path.join(folder, 'day_%d.parquet' % day)
    df.write.parquet(path, mode=mode)


def load_transformed(spark, folder, days):
    reader = spark.read.parquet
    paths = [os.path.join(folder, 'day_%d.parquet' % i) for i in range(days)]
    return reader(*paths), [reader(path) for path in paths]


def save_final(dfs, folder, mode=None):
    for day, df in enumerate(dfs):
        path = os.path.join(folder, 'day_%d.final' % day)
        df.write.parquet(path, mode=mode)


def load_combined_model(spark, model_folder):
    path = os.path.join(model_folder, 'combined.parquet')
    return spark.read.parquet(path)


def save_combined_model(df, model_folder, mode=None):
    path = os.path.join(model_folder, 'combined.parquet')
    df.write.parquet(path, mode=mode)


def delete_combined_model(spark, model_folder):
    path = os.path.join(model_folder, 'combined.parquet')
    delete_data_source(spark, path)


def load_column_models(spark, model_folder):
    for i in CAT_COLS:
        path = os.path.join(model_folder, '%d.parquet' % i)
        yield i, spark.read.parquet(path)


def save_column_models(column_models, model_folder, mode=None):
    for i, model in column_models:
        path = os.path.join(model_folder, '%d.parquet' % i)
        model.write.parquet(path, mode=mode)


_benchmark = {}


@contextmanager
def _timed(step):
    start = time()
    yield
    end = time()
    _benchmark[step] = end - start


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument('--days', type=int, required=True)
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--model_folder', required=True)
    parser.add_argument(
        '--write_mode',
        choices=['overwrite', 'errorifexists'],
        default='errorifexists')

    parser.add_argument('--frequency_limit')
    parser.add_argument('--transform_log', action='store_true')
    parser.add_argument('--broadcast_model', action='store_true')
    parser.add_argument('--randomize', choices=['total', 'day'])

    parser.add_argument('--debug_mode', action='store_true')

    return parser.parse_args()


def _main():
    args = _parse_args()
    spark = SparkSession.builder.getOrCreate()

    df_all, dfs = load_raw(spark, args.input_folder, args.days)

    with _timed('generate dicts'):
        save_combined_model(
            get_combined_model(df_all, args.frequency_limit),
            args.model_folder,
            args.write_mode)
        save_column_models(
            get_column_models(load_combined_model(spark, args.model_folder)),
            args.model_folder,
            args.write_mode)
        if not args.debug_mode:
            delete_combined_model(spark, args.model_folder)

    transformed_folder = 'transformed'

    with _timed('transform'):
        models = list(load_column_models(spark, args.model_folder))
        for day, df in enumerate(dfs):
            df_encoded = apply_models(df, models, args.broadcast_model)
            df_transformed = transform_log(df_encoded, args.transform_log)
            save_transformed(df_transformed, transformed_folder, day, args.write_mode)

    df_all, dfs = load_transformed(spark, transformed_folder, args.days)
    counts = [df.count() for df in dfs]

    if not args.randomize:
        # TODO: upgrade the logic here
        save_final(dfs, args.output_folder, args.write_mode)

    if args.randomize == 'total':
        splits = split(randomize(df_all), counts)
        save_final(splits, args.output_folder, args.write_mode)

    if args.randomize == 'day':
        splits = (randomize(df) for df in dfs)
        save_final(splits, args.output_folder, args.write_mode)

    print('=' * 100)
    print(_benchmark)


if __name__ == '__main__':
    _main()
