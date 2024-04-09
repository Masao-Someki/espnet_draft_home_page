#!/usr/bin/env sh

# abort on errors
# set -e

# First move the documents into docs directory
documents=("espnet2" "tutorials" "espnetez")

cd docs

# convert notebooks into markdown
find ./docs \
    -type f \
    -name '*.ipynb' \
    -exec bash -c "jupyter nbconvert --clear-output \"{}\"" \;

find ./docs \
    -type f \
    -name '*.ipynb' \
    -exec bash -c "jupyter nbconvert --to markdown \"{}\"" \;

cd -

# create meny configs
python create_menu.py --dlist espnet2 espnetez tutorials

# build
npm run docs:build

# copy files
mkdir -p dist
cp docs/.vuepress/dist dist/
