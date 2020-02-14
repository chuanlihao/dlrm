SPARK_MASTER=spark://$HOSTNAME:7077
DRIVER_MEMORY=16G
EXECUTOR_MEMORY=42G
EXECUTOR_CORES=10
SHUFFLE_PARTITIONS=$(($EXECUTOR_CORES * 2))

INPUT_PATH=day_0.csv
OUTPUT_PATH=day_0_encoded.parquet
MODEL_FOLDER=models
FREQUENCY_LIMIT=1
# 0 or 1
BROADCAST_MODEL=1
# 0 or 1
TRANSFORM_LOG=1

# errorifexists or overwrite, set 'errorifexists' for production
WRITE_MODE=overwrite
# 0 or 1, disable it for production
DEBUG_MODE=1

OPTS=""
if [[ $FREQUENCY_LIMIT != "0" && $FREQUENCY_LIMIT != "1" ]]; then
    OPTS="$OPTS --frequency_limit $FREQUENCY_LIMIT"
fi
if [[ $BROADCAST_MODEL == 1 ]]; then
    OPTS="$OPTS --broadcast_model"
fi
if [[ $TRANSFORM_LOG == 1 ]]; then
    OPTS="$OPTS --transform_log"
fi
if [[ $DEBUG_MODE == 1 ]]; then
    OPTS="$OPTS --debug_mode"
fi

spark-submit \
    --master $SPARK_MASTER \
    --driver-memory $DRIVER_MEMORY \
    --executor-memory $EXECUTOR_MEMORY \
    --num-executors 1 \
    --executor-cores $EXECUTOR_CORES \
    --conf spark.task.cpus=1 \
    --conf spark.sql.files.maxPartitionBytes=1G \
    --conf spark.sql.shuffle.partitions=$SHUFFLE_PARTITIONS \
    spark_data_utils.py \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --model_folder $MODEL_FOLDER \
    --write_mode $WRITE_MODE \
    $OPTS
