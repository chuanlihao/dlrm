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
from operator import itemgetter
from time import time

from pyspark import broadcast
from pyspark.sql import Row, SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *


LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))


def get_column_counts_with_frequency_limit(df, frequency_limit = None):
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
    return df

def assign_id_with_window(df):
    windowed = Window.partitionBy('column_id').orderBy(desc('count'))
    return (df
            .withColumn('id', row_number().over(windowed))
            .drop('count'))

def assign_low_mem_partial_ids(df):
    # To avoid some scaling issues with a simple window operation, we use a more complex method
    # to compute the same thing, but in a more distributed spark specific way
    df = df.orderBy(asc('column_id'), desc('count'))
    # The monotonically_increasing_id is the partition id in the top 31 bits and the rest
    # is an increasing count of the rows within that partition.  So we split it into two parts,
    # the partion id part_id and the count mono_id
    df = df.withColumn('part_id', spark_partition_id())
    return df.withColumn('mono_id', monotonically_increasing_id() - shiftLeft(col('part_id'), 33))

def assign_low_mem_final_ids(df):
    # Now we can find the minimum and maximum mono_ids within a given column/partition pair
    sub_model = df.groupBy('column_id', 'part_id').agg(max('mono_id').alias('top'), min('mono_id').alias('bottom'))
    sub_model = sub_model.withColumn('diff', col('top') - col('bottom') + 1)
    sub_model = sub_model.drop('top')
    # This window function is over aggregated column/partition pair table. It will do a running sum of the rows
    # within that column
    windowed = Window.partitionBy('column_id').orderBy('part_id').rowsBetween(Window.unboundedPreceding, -1)
    sub_model = sub_model.withColumn('running_sum', sum('diff').over(windowed)).na.fill(0, ["running_sum"])

    joined = df.withColumnRenamed('column_id', 'i_column_id')
    joined = joined.withColumnRenamed('part_id', 'i_part_id')

    # Then we can join the original input with the pair it is a part of
    joined = joined.join(sub_model, (col('i_column_id') == col('column_id')) & (col('part_id') == col('i_part_id')))

    # So with all that we can subtract bottom from mono_id makeing it start at 0 for each partition
    # and then add in the running_sum so the id is contiguous and unique for the entire column. + 1 to make it match the 1 based indexing
    # for row_number
    ret = joined.select(col('column_id'), col('data'), (col('mono_id') - col('bottom') + col('running_sum') + 1).cast(IntegerType()).alias('id'))
    return ret

def get_column_models(combined_model):
    for i in CAT_COLS:
        model = (combined_model
            .filter('column_id == %d' % (i - CAT_COLS[0]))
            .drop('column_id'))
        yield i, model


def apply_models(df, models, broadcast_model = False):
    if not broadcast_model:
        # sort the models smallest to largest so
        # we reduce the amount of shuffle data sooner than later
        models = sorted(models, key=itemgetter(2))
    for i, model, size in models:
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

def rand_ordinal(df):
    # create a random long from the double precision float.  
    # The fraction part of a double is 52 bits, so we try to capture as much
    # of that as possible
    return df.withColumn('ordinal', (rand() * (1 << 52)).cast(LongType()))

def day_from_ordinal(df, num_days):
    return df.withColumn('day', (col('ordinal') % num_days).cast(IntegerType()))

def day_from_input_file(df):
    return df.withColumn('day', substring_index(input_file_name(), '_', -1).cast(IntegerType()))

def psudo_sort_by_day_plus(spark, df, num_days):
    # Sort is very expensive because it needs to calculate the partitions
    # which in our case may involve rereading all of the data.  In some cases
    # we can avoid this by repartitioning the data and sorting within a single partition
    shuffle_parts = int(spark.conf.get('spark.sql.shuffle.partitions'))
    extra_parts = int(shuffle_parts/num_days)
    if extra_parts <= 0:
        df = df.repartition('day')
    else:
        #We want to spread out the computation to about the same amount as shuffle_parts
        divided = (col('ordinal') / num_days).cast(LongType())
        extra_ident = divided % extra_parts
        df = df.repartition(col('day'), extra_ident)
    return df.sortWithinPartitions('day', 'ordinal')


def load_combined_model(spark, model_folder):
    path = os.path.join(model_folder, 'combined.parquet')
    return spark.read.parquet(path)


def save_combined_model(df, model_folder, mode=None):
    path = os.path.join(model_folder, 'combined.parquet')
    df.write.parquet(path, mode=mode)


def delete_combined_model(spark, model_folder):
    path = os.path.join(model_folder, 'combined.parquet')
    delete_data_source(spark, path)


def load_low_mem_partial_ids(spark, model_folder):
    path = os.path.join(model_folder, 'partial_ids.parquet')
    return spark.read.parquet(path)


def save_low_mem_partial_ids(df, model_folder, mode=None):
    path = os.path.join(model_folder, 'partial_ids.parquet')
    df.write.parquet(path, mode=mode)


def delete_low_mem_partial_ids(spark, model_folder):
    path = os.path.join(model_folder, 'partial_ids.parquet')
    delete_data_source(spark, path)


def load_column_models(spark, model_folder):
    for i in CAT_COLS:
        path = os.path.join(model_folder, '%d.parquet' % i)
        df = spark.read.parquet(path)
        yield i, df, df.count()

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
    #Support for running in a lower memory environment
    parser.add_argument('--low_mem', action='store_true')
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
        col_counts = get_column_counts_with_frequency_limit(df, args.frequency_limit)
        if args.low_mem:
            # in low memory mode we have to save an intermediate result
            # because if we try to do it in one query spark ends up assigning the
            # partial ids in two different locations that are not guaranteed to line up
            # this prevents that from happening by assigning the partial ids
            # and then writeing them out.
            save_low_mem_partial_ids(
                    assign_low_mem_partial_ids(col_counts),
                    args.model_folder,
                    args.write_mode)
            save_combined_model(
                    assign_low_mem_final_ids(load_low_mem_partial_ids(spark, args.model_folder)),
                    args.model_folder,
                    args.write_mode)
            if not args.debug_mode:
                delete_low_mem_partial_ids(spark, args.model_folder)

        else:
            save_combined_model(
                    assign_id_with_window(col_counts),
                    args.model_folder,
                    args.write_mode)
        save_column_models(
            get_column_models(load_combined_model(spark, args.model_folder)),
            args.model_folder,
            args.write_mode)
        if not args.debug_mode:
            delete_combined_model(spark, args.model_folder)

    with _timed('transform and shuffle'):
        if args.output_ordering == 'total_random':
            df = rand_ordinal(df)
            if args.output_partitioning == 'day':
                df = day_from_ordinal(df, args.days)
        elif args.output_ordering == 'day_random':
            df = rand_ordinal(df)
            df = day_from_input_file(df)
        elif args.output_ordering == 'input':
            df = df.withColumn('ordinal', monotonically_increasing_id())
            if args.output_partitioning == 'day':
                df = day_from_input_file(df)
        else: # any ordering
            if args.output_partitioning == 'day':
                df = day_from_input_file(df)

        
        df = apply_models(
            df,
            load_column_models(spark, args.model_folder),
            not args.low_mem)
        df = transform_log(df, not args.no_numeric_log_col)


        if args.output_partitioning == 'day':
            partitionBy = 'day'
        else:
            partitionBy = None

        if args.output_ordering == 'total_random':
            if args.output_partitioning == 'day':
                df = psudo_sort_by_day_plus(spark, df, args.days)
            else: # none
                # Don't do a full sort it is expensive. Order is random so
                # just make it random
                df = df.repartition('ordinal').sortWithinPartitions('ordinal')

            df = df.drop('ordinal')
        elif args.output_ordering == 'day_random':
            df = psudo_sort_by_day_plus(spark, df, args.days)
            df = df.drop('ordinal')
            if args.output_partitioning != 'day':
                df = df.drop('day')
        elif args.output_ordering == 'input':
            if args.low_mem:
                # This is the slowest option. We totally messed up the order so we have to put
                # it back in the correct order
                df = df.orderBy('ordinal')
            else:
                # Applying the dictionary happened within a single task so we are already really
                # close to the correct order, just need to sort within the partition
                df = df.sortWithinPartitions('ordinal')
            df = df.drop('ordinal')
            if args.output_partitioning != 'day':
                df = df.drop('day')
        # else: any ordering so do nothing the ordering does not matter

        df.write.parquet(
            args.output_folder,
            mode=args.write_mode,
            partitionBy=partitionBy)

    print('=' * 100)
    print(_benchmark)


if __name__ == '__main__':
    _main()
