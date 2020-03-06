SPARK_MASTER=spark://$HOSTNAME:7077
DRIVER_MEMORY=8G
NUM_EXECUTORS=1
EXECUTOR_MEMORY=32G
EXECUTOR_CORES=8

INPUT_FOLDER=clicklog
DAYS=0-22
OUTPUT_FOLDER=final
MODEL_FOLDER=models

# low memory mode
LOW_MEM=1

OPTS=""
if [[ $LOW_MEM == 1 ]]; then
    OPTS="$OPTS --low_mem"
fi

spark-submit \
    --master $SPARK_MASTER \
    --driver-memory $DRIVER_MEMORY \
    --executor-memory $EXECUTOR_MEMORY \
    --num-executors $NUM_EXECUTORS \
    --executor-cores $EXECUTOR_CORES \
    --conf spark.task.cpus=1 \
    --conf spark.sql.files.maxPartitionBytes=1G \
    spark_data_utils.py \
    --mode transform \
    --input_folder $INPUT_FOLDER \
    --days $DAYS \
    --output_folder $OUTPUT_FOLDER \
    --model_folder $MODEL_FOLDER \
    $OPTS
