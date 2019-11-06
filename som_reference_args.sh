#!/bin/sh
DATA=/data/flowcat-data/mll-flowdata/decCLL-9F
META=/data/flowcat-data/mll-flowdata/decCLL-9F.2019-10-29.meta/train.json.gz
LABELS=/data/flowcat-data/2019-10_paper_data/0-final/dataset/references.json
OUTPUT=/data/flowcat-data/2019-11_paper_additonal/diff_epochs

format_config() {
    echo "{
    \"max_epochs\": $1,
    \"initial_radius\": $2,
    \"end_radius\": $3,
    \"radius_cooling\": \"linear\",
    \"map_type\": \"toroid\",
    \"dims\": [32, 32, -1],
    \"scaler\": \"MinMaxScaler\"
    }"
}

runs="ep1;1
ep2;2
ep4;4
ep8;8
ep10;10
ep20;20
ep40;40"

_IFS=$IFS

IFS="
"
for run in $runs; do
    echo $run | (
       IFS=";" read name epochs
       echo "Training $name with $epochs epochs"
       config=$(format_config $epochs 16 2)
       for i in $(seq 1 10); do
       	   echo $i
       	   flowcat reference --data "$DATA" --meta "$META" --labels "$LABELS" --output "$OUTPUT/$name/rep$i" --tensorboard 1 --trainargs "$config"
       done
    )
done
IFS=_IFS


