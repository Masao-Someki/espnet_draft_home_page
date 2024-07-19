#!/usr/bin/env bash

set -euo pipefail

if [ ! -d ./doc/notebook ]; then
    cd ./doc
    git clone https://github.com/espnet/notebook --depth 1
    cd ..
fi

echo "#  Notebook

Jupyter notebooks for course demos and tutorials.
"

documents=("ESPnet1" "ESPnet2")
# documents=("ESPnet1")
for document in "${documents[@]}"; do
    echo "## ${document}
"
    find ./doc/notebooks/${document} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c "jupyter nbconvert --clear-output \"{}\"" \;
    find ./doc/notebooks/${document} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c "jupyter nbconvert --to markdown \"{}\"" \;
    
    for md_file in `find "./doc/notebooks/${document}" -name "*.md"`; do
        filename=`basename ${md_file}`
        echo "* [${filename}](./${md_file:11:100})"
    done
    for ipynb_file in `find "./doc/notebooks/${document}" -name "*.ipynb"`; do
        rm ${ipynb_file}
    done
done
