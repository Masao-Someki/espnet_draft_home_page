# ESPnet home page (draft)

This is the draft version of ESPnet home page.

This project use [VuePress2](https://v2.vuepress.vuejs.org/).

## Project setup

1. install nodejs>=18.16.0
   1. install node.js and npm
   ```shell
   apt install -u nodejs npm
   ```
   2. install n for package management
   ```shell
   npm instal n -g
   ```
   3. install latest nodejs and npm
   ```shell
   n stable
   ```
   4. uninstall the nodejs and npm we install at the first step.
   ```shell
   apt purge-y nodejs npm
   apt autoremove -y
   ```

2. install npm packages
```shell
npm i
```

3. install jupyter and nbconvert command
```shell
# install jupyter
pip install jupyter

# install nbconvert
## You need pandoc
apt-get install pandoc
git clone https://github.com/jupyter/nbconvert.git
cd nbconvert
pip install -e .
```

## Run debug server and build application

- Run debug server: If there is no menubars.yml or sidebars.yml, or if you have created a new markdown file, then run `create_menu.py` before the following command.
```shell
npm run docs:dev
```

- Build configuration for menubar/sidebar
```shell
python create_menu.py \
    --dlist espnet2 espnetez ...
```

- Build webpage
```shell
# Build pages including ipynb->markdown conversion
. deploy.sh

# Only build webpage from markdown
npm run docs:build
```

## How to edit

All markdown files under `docs` directory will be automatically converted into webpages.
Markdown files named `readme.md` or `index.md` will be considered as the landing page for that directory.

So for example, the following espnetez asr directory has the following three pages and no landing page.

**docs/espnetez**
```
espnetez/
├── asr
│   ├── finetune_owsm.md
│   ├── finetune_with_lora.md
│   └── train.md
└── tts
    └── tacotron2.md
```

**pages**
- `localhost:8080/espnet_page/espnetez/asr/finetune_owsm`
- `localhost:8080/espnet_page/espnetez/asr/finetune_with_lora`
- `localhost:8080/espnet_page/espnetez/asr/train`


