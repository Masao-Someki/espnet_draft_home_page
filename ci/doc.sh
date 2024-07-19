#!/usr/bin/env bash

. ./tools/activate_python.sh

build_and_convert () {
    # $1: path
    # $2: output
    mkdir -p ./doc/_gen/$2
    echo "# $2

" > ./doc/_gen/$2.md
    for filename in `find $1`; do
        bn=`basename ${filename}`
        echo "Converting ${filename} to rst..."
        ./doc/usage2rst.sh ${filename} > ./doc/_gen/tools/$2/${bn}.rst
        echo "- [${bn}](./tools/$2/${bn}.md)" >> ./doc/_gen/$2.md
    done
}

if [ ! -e tools/kaldi ]; then
    git clone https://github.com/kaldi-asr/kaldi --depth 1 tools/kaldi
fi

# build sphinx document under doc/
mkdir -p doc/_gen
mkdir -p doc/_gen/tools

# NOTE allow unbound variable (-u) inside kaldi scripts
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}
# set -euo pipefail

# generate tools doc
./doc/argparse2rst.py \
    --title utils_py \
    --output_dir utils_py \
    ./utils/*.py > ./doc/_gen/utils_py.rst
mv utils_py ./doc/_gen/tools

./doc/argparse2rst.py \
    --title espnet_bin \
    --output_dir espnet_bin \
    ./espnet/bin/*.py > ./doc/_gen/espnet_bin.rst
mv espnet_bin ./doc/_gen/tools

./doc/argparse2rst.py \
    --title espnet2_bin \
    --output_dir espnet2_bin \
    ./espnet2/bin/*.py > ./doc/_gen/espnet2_bin.rst
mv espnet2_bin ./doc/_gen/tools

build_and_convert "utils/*.sh" utils_sh
build_and_convert "tools/sentencepiece_commands/spm_*" spm

./doc/notebook2rst.sh > ./doc/_gen/notebooks.md

# generate package doc
./doc/members2rst.py --root espnet --dst ./doc/_gen/packages --exclude espnet.bin
./doc/members2rst.py --root espnet2 --dst ./doc/_gen/packages --exclude espnet2.bin
./doc/members2rst.py --root espnetez --dst ./doc/_gen/packages


# build html
# TODO(karita): add -W to turn warnings into errors
cp ./doc/index.rst ./doc/_gen/index.rst
cp ./doc/conf.py ./doc/_gen/
rm ./doc/_gen/tools/espnet2_bin/*_train.rst
sphinx-build -M markdown ./doc/_gen ./doc/build

cp -r ./doc/build/markdown/* ./doc/vuepress/docs/
cp -r ./doc/notebook ./doc/vuepress/docs/
cp ./doc/*.md ./doc/vuepress/docs/
cp -r ./doc/image ./doc/vuepress/docs/

find ./doc/vuepress/docs/ -name "*.md" -exec sed -i 's/^> - \[/- \[/g' {} \;
find ./doc/vuepress/docs/ -name "*.md" -exec sed -i 's/```default/```text/g' {} \;
find ./doc/vuepress/docs/ -name "*.md" -exec sed -i 's/```pycon/```python/g' {} \;
./doc/convert_custom_tags_to_html.py ./doc/vuepress/docs/

python ./doc/vuepress/create_menu.py \
    --root ./doc/vuepress/docs \
    --dlist tools packages notebook

# check if node is installed
if which node > /dev/null
then
    echo "node is installed, skipping..."
else
    apt install -y nodejs npm
    npm install n -g
    n stable
    apt purge -y nodejs npm
    apt autoremove -y
fi

cd ./doc/vuepress
npm i
npm run docs:build

mv .vuepress/dist ../