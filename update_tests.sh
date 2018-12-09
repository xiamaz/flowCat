#!/bin/sh

TEST_FOLDER="tests/data"

down() {
	s3url="$1"
	echo "Downloading test data from $s3url"
	aws s3 sync $s3url $TEST_FOLDER
}

up() {
	s3url="$1"
	echo "Uploading test data to $s3url"
	aws s3 sync $TEST_FOLDER $s3url
}

sync() {
	down "$1"
	up "$1"
}

if [ -z "$2" ]; then
	s3url="s3://flowcat-test"
	echo "Using default s3 bucket $s3url"
else
	s3url="$2"
fi

case "$1" in
down|up|sync)
		$1 "$s3url"
		;;
	'')
		cat <<EOF
Usage: CMD (s3url - default: s3://flowcat-test)
    down - Download all data from S3 repository
    up - Upload all data to S3 repository
    sync - Download all data and upload all new data
EOF
		;;
esac
