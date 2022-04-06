#!/bin/bash
pip install black==22.3.0
arrVar=()
echo we ignore non-*.py files and files generated from protobuf
excluded_files=(
   docarray/proto/docarray_pb2.py
   docs/conf.py
)
for changed_file in $CHANGED_FILES; do
  if [[ ${changed_file} == *.py ]] && ! [[ " ${excluded_files[@]} " =~ " ${changed_file} " ]]; then
    echo checking ${changed_file}
    arrVar+=(${changed_file})
  fi
done
if [ ${#arrVar[@]} -ne 0 ]; then
  black -S --check "${arrVar[@]}"
fi
