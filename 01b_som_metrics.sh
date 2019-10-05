#!/bin/sh
calculate_metric() {
    ./01b_calculate_som_metrics.py --fcsdata output/4-flowsom-cmp/samples --fcsmeta output/4-flowsom-cmp/samples/samples.json --somdata $1 --output output/4-flowsom-cmp/quantization-error/$(basename $1)
}
mkdir -p output/4-flowsom-cmp/quantization-error
paths="output/4-flowsom-cmp/flowsom-10;output/4-flowsom-cmp/flowcat-refit-s10;output/4-flowsom-cmp/flowsom-32;output/4-flowsom-cmp/flowcat-refit-s32;output/4-flowsom-cmp/flowcat-denovo"
sep=$IFS
IFS=';'
for path in $paths; do
    echo $path
    calculate_metric "$path"
done
