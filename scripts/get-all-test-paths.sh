#!/usr/bin/env bash

set -ex

BATCH_SIZE=3
#declare -a array1=( 'tests/unit/test_*.py' )
#declare -a array2=( $(ls -d tests/unit/*/ | grep -v '__pycache__' | grep -v 'array') )
#declare -a array3=( 'tests/unit/array/*.py' )
declare -a mixins=( $(find tests -name 'test_*.py' | grep -v 'test_tensorrt.py') )
declare -a array4=( '$(echo '${mixins[@]}' | xargs -n$BATCH_SIZE)' )
# array5 is currently empty because in the array/ directory, mixins is the only directory
# but add the following in case new directories are created in array/
declare -a array5=( $(ls -d tests/unit/array/*/ | grep -v '__pycache__' | grep -v 'mixins') )
dest=( '${array1[@]}' '${array2[@]}' '${array3[@]}' '${array4[@]}' '${array5[@]}' )

printf '%s\n' '${dest[@]}' | jq -R . | jq -cs .
