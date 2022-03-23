#!/usr/bin/env bash

set -ex

rm -rf api && make clean

make dirhtml
