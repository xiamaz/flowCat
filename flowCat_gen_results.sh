#!/bin/sh
DATA=/data/flowcat-data/mll-flowdata/decCLL-9F
META=/data/flowcat-data/mll-flowdata/decCLL-9F.2019-10-29.meta/train.json.gz
META_TEST="/data/flowcat-data/mll-flowdata/decCLL-9F.2019-10-29.meta/test.json.gz"
LABELS=/data/flowcat-data/mll-flowdata/decCLL-9F.2019-10-29.meta/references.json

OUTPUT=/data/flowcat-data/paper-cytometry
# I. Create reference SOM
REF_OUTPUT="$OUTPUT/reference"
CONFIG="{
    \"max_epochs\": 20,
    \"initial_radius\": 16,
    \"end_radius\": 2,
    \"radius_cooling\": \"linear\",
    \"map_type\": \"toroid\",
    \"dims\": [32, 32, -1],
    \"scaler\": \"MinMaxScaler\"
}"
if [ ! -d $REF_OUTPUT ]; then
    flowcat reference --data "$DATA" --meta "$META" --labels "$LABELS" --output "$REF_OUTPUT" --tensorboard 1 --trainargs "$CONFIG"
else
    echo "Ref SOM found in $REF_OUTPUT. Skipping..."
fi

# II. Transform all data using the reference SOM
SOM_OUTPUT="$OUTPUT/som/train"
if [ ! -d $SOM_OUTPUT ]; then
    flowcat transform --data "$DATA" --meta "$META" --reference "$REF_OUTPUT" --output $SOM_OUTPUT
else
    echo "Transformed SOM found in $SOM_OUTPUT. Skipping..."
fi

SOM_OUTPUT_TEST="$OUTPUT/som/test"
if [ ! -d $SOM_OUTPUT_TEST ]; then
    flowcat transform --data "$DATA" --meta "$META_TEST" --reference "$REF_OUTPUT" --output $SOM_OUTPUT_TEST
else
    echo "Transformed SOM found in $SOM_OUTPUT_TEST. Skipping..."
fi
