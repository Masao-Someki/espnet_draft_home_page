#!/usr/bin/env bash

# set -euo pipefail

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
    echo "## ${document}\n"
    find ./doc/notebook/${document} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c ". tools/activate_python.sh;jupyter nbconvert --clear-output \"{}\"" \;
    find ./doc/notebook/${document} \
        -type f \
        -name '*.ipynb' \
        -exec bash -c ". tools/activate_python.sh;jupyter nbconvert --to markdown \"{}\"" \;
    
    basedir=./doc/notebook/${document}
    for md_file in `find "./doc/notebook/${document}" -name "*.md"`; do
        filename=`basename ${md_file}`
        echo "* [${filename}](./${md_file:((${#basedir}+1)):100})"
    done
    for ipynb_file in `find "./doc/notebook/${document}" -name "*.ipynb"`; do
        rm ${ipynb_file}
    done

    # generate README.md
    echo "# ${document} Demo" > ./doc/notebook/${document}/README.md
    for md_file in `find "./doc/notebook/${document}" -name "*.md"`; do
        filename=`basename ${md_file}`
        echo "* [${filename}](./${md_file:((${#basedir}+1)):100})" >> ./doc/notebook/${document}/README.md
    done

    echo ""

done
