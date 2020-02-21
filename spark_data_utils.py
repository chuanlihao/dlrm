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
    paths = [os.path.join(folder, 'day_%d' % i) for i in range(days)]
    return (spark
        .read
        .schema(schema)
        .option('sep', '\t')
        .csv(paths))


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
    parser.add_argument('--no_numeric_log_col', action='store_true')
    parser.add_argument('--no_model_broadcast', action='store_true')
    parser.add_argument(
        '--output_ordering',
        choices=['total_random', 'day_random', 'any', 'input'],
        default='total_random')

    parser.add_argument(
        '--output_partitioning',
        choices=['day', 'none'],
        default='none')

    parser.add_argument('--debug_mode', action='store_true')

    return parser.parse_args()


def _main():
    args = _parse_args()
    spark = SparkSession.builder.getOrCreate()

    df = load_raw(spark, args.input_folder, args.days)

    with _timed('generate dicts'):
        save_combined_model(
            get_combined_model(df, args.frequency_limit),
            args.model_folder,
            args.write_mode)
        save_column_models(
            get_column_models(load_combined_model(spark, args.model_folder)),
            args.model_folder,
            args.write_mode)
        if not args.debug_mode:
            delete_combined_model(spark, args.model_folder)

    with _timed('transform and shuffle'):
        if args.output_ordering == 'day_random' or args.output_partitioning == 'day':
            if args.output_ordering == 'total_random':
                df = df.withColumn('day', (rand() * args.days).cast(IntegerType()))
            else:
                df = df.withColumn('day', input_file_name()) #TODO need to make this smaller

        if args.output_ordering == 'day_random' or args.output_ordering == 'total_random':
            df = df.withColumn('ordinal', rand())
        elif args.output_ordering == 'input':
            df = df.withColumn('ordinal', monotonically_increasing_id())

        df = apply_models(
            df,
            load_column_models(spark, args.model_folder),
            not args.no_model_broadcast)
        df = transform_log(df, not args.no_numeric_log_col)

        if args.output_ordering != 'any':
            if any ('day' in c for c in df.columns):
                df = df.orderBy('day', 'ordinal')
            else:
                df = df.orderBy('ordinal')
            df = df.drop('ordinal')

        if args.output_partitioning == 'day':
            partitionBy = 'day'
        else:
            partitionBy = None

        df.write.parquet(
            args.output_folder,
            mode=args.write_mode,
            partitionBy=partitionBy)

    print('=' * 100)
    print(_benchmark)


if __name__ == '__main__':
    _main()
