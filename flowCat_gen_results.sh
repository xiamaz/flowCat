#!/bin/sh
set -e
set -u
DATA=/data/flowcat-data/mll-flowdata/decCLL-9F
META=/data/flowcat-data/mll-flowdata/decCLL-9F.2019-10-29.meta/train.json.gz
META_TEST="/data/flowcat-data/mll-flowdata/decCLL-9F.2019-10-29.meta/test.json.gz"
LABELS=/data/flowcat-data/mll-flowdata/decCLL-9F.2019-10-29.meta/references.json

DATA_UNUSED=/data/flowcat-data/paper-cytometry/unused-data/data
META_UNUSED=/data/flowcat-data/paper-cytometry/unused-data/meta.json.gz

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
    echo "Transformed test SOM found in $SOM_OUTPUT_TEST. Skipping..."
fi

SOM_OUTPUT_UNUSED="$OUTPUT/som/unused"
if [ ! -d $SOM_OUTPUT_UNUSED ]; then
    flowcat transform --data "$DATA_UNUSED" --meta "$META_UNUSED" --reference "$REF_OUTPUT" --output $SOM_OUTPUT_UNUSED
else
    echo "Transformed untrained SOM found in $SOM_OUTPUT_UNUSED. Skipping..."
fi

# III. Train a model on the training data with additional validation
# information.

MODEL_OUTPUT="$OUTPUT/classifier"
if [ ! -d $MODEL_OUTPUT ]; then
    flowcat train --data "$SOM_OUTPUT" --output "$MODEL_OUTPUT"
    flowcat predict --data "$SOM_OUTPUT" --model "$MODEL_OUTPUT" --output "$MODEL_OUTPUT" --labels "$MODEL_OUTPUT/ids_validate.json"
else
    echo "Trained model found at $MODEL_OUTPUT. Skipping..."
fi

# IV. Test data predictions
TEST_OUTPUT="$OUTPUT/testset"
if [ ! -d $TEST_OUTPUT ]; then
    flowcat predict --data "$SOM_OUTPUT_TEST" --model "$MODEL_OUTPUT" --output "$TEST_OUTPUT" --metrics 1
else
    echo "Testset predictions already found at $TEST_OUTPUT. Skipping..."
fi

TSNE_OUTPUT="$OUTPUT/tsne"
if [ ! -d $TSNE_OUTPUT ]; then
    flowcat predict --data "$SOM_OUTPUT_UNUSED" --model "$MODEL_OUTPUT" --output "$TSNE_OUTPUT" --metrics 0
else
    echo "TSNE predictions already found at $TSNE_OUTPUT. Skipping..."
fi
