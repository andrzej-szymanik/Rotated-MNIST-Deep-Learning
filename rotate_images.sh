#!/bin/bash -e

CUR_DIR=`pwd`
cd "${1}"

    for ((i=0; i<15000; i++)) do
        for file in *.jpg; do
            convert "${file}" -rotate 90 "${file}";
        done
    done
cd CUR_DIR 
