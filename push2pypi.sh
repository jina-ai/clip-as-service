#!/usr/bin/env bash

CLIENT_CODE='client/bert_serving/client/__init__.py'
SERVER_CODE='client/bert_serving/server/__init__.py'
VER_TAG='__version__ = '

VER=$(grep "$VER_TAG" $CLIENT_CODE | sed -n 's/^.*'\''\([^'\'']*\)'\''.*$/\1/p')
echo 'current version: '$VER

VER=$(echo $VER | awk -F. -v OFS=. 'NF==1{print ++$NF}; NF>1{if(length($NF+1)>length($NF))$(NF-1)++; $NF=sprintf("%0*d", length($NF), ($NF+1)%(10^length($NF))); print}')
echo 'increased version: '$VER

VER_LINE="$VER_TAG '$VER'"
echo $VER_LINE