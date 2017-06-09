#! /bin/bash

DIR="./data"

if [ ! -e ${DIR} ] ; then
	mkdir -p ${DIR}
fi

URL="http://yann.lecun.com/exdb/mnist"
FILES=("train-images-idx3-ubyte.gz" "t10k-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" "t10k-labels-idx1-ubyte.gz")
for filename in ${FILES[@]}; do
	wget -nc -P ${DIR} "${URL}/${filename}"
	gzip -d "${DIR}/${filename}"
done
