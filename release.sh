#!/usr/bin/env bash

set -e

function escape_slashes {
    sed 's/\//\\\//g'
}

function change_line {
    local OLD_LINE_PATTERN=$1
    local NEW_LINE=$2
    local FILE=$3

    local NEW=$(echo "${NEW_LINE}" | escape_slashes)
    sed -i .bak '/'"${OLD_LINE_PATTERN}"'/s/.*/'"${NEW}"'/' "${FILE}"
    mv "${FILE}.bak" /tmp/
}

function clean_build {
    rm -rf dist
    rm -rf *.egg-info
    rm -rf build
}

function pub_pypi {
    # publish to pypi
    cp README.md $1
    cd $1
    clean_build
    python setup.py sdist bdist_wheel $2
    twine upload dist/*
    clean_build
    cd -
}

CLIENT_DIR='client/'
SERVER_DIR='server/'
CLIENT_CODE=$CLIENT_DIR'bert_serving/client/__init__.py'
SERVER_CODE=$SERVER_DIR'bert_serving/server/__init__.py'
CLIENT_MD=$CLIENT_DIR'README.md'
SERVER_MD=$SERVER_DIR'README.md'
VER_TAG='__version__ = '

#$(grep "$VER_TAG" $CLIENT_CODE | sed -n 's/^.*'\''\([^'\'']*\)'\''.*$/\1/p')
VER=$(git tag -l |tail -n1)
echo 'current version: '$VER

VER=$(echo $VER | awk -F. -v OFS=. 'NF==1{print ++$NF}; NF>1{if(length($NF+1)>length($NF))$(NF-1)++; $NF=sprintf("%0*d", length($NF), ($NF+1)%(10^length($NF))); print}')
echo 'increased to version: '$VER

# write back tag to client and server code
VER_VAL=$VER_TAG"'"${VER#"v"}"'"

change_line "$VER_TAG" "$VER_VAL" $CLIENT_CODE
change_line "$VER_TAG" "$VER_VAL" $SERVER_CODE
git add $CLIENT_CODE $SERVER_CODE $CLIENT_MD $SERVER_MD
git commit -m 'increase version number'
git push origin master
git tag $VER
git push -u origin --tags

pub_pypi $SERVER_DIR
pub_pypi $CLIENT_DIR  '--universal'
