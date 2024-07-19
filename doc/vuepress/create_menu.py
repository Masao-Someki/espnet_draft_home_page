import os
import glob
import argparse
import yaml


def get_menubar_recursively(directory):
    menubars = [{
        'text': directory.lower().split('/')[-1].upper(),
        'children': []
    }]
    print(f'Scanning directory: {directory}')
    for child in glob.glob(os.path.join(directory, '*')):
        if os.path.isdir(child):
            children = get_menubar_recursively(child)
            if len(children) > 0:
                menubars[0]['children'].append(children)
                # menubars.append({
                #     'text': child.lower().split('/')[-1].upper(),
                #     'children': get_menubar_recursively(child)
                # })
        else:
            if os.path.splitext(child)[1].lower() == '.ipynb':
                menubars.append(f'/{child[len(DOCS):-6]}') # remoce '.ipynb'
            elif os.path.splitext(child)[1].lower() == '.md':
                menubars.append(f'/{child[len(DOCS):-3]}') # remoce '.ipynb'
    return menubars


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        type=str, help='List of directory')
    parser.add_argument('--dlist', required=True, nargs="*",
                        type=str, help='List of directory')
    args = parser.parse_args()

    menubars = []
    sidebars = {}

    DOCS = args.root + '/'

    # add links
    menubars.append({
        'text': 'Github',
        'link': 'https://github.com/espnet/espnet'
    })
    menubars.append({
        'text': 'HuggingFace',
        'link': 'https://huggingface.co/espnet'
    })
    # menubars.append({
    #     'text': 'Docs',
    #     'link': 'https://espnet.github.io/espnet'
    # })

    for directory in args.dlist:
        menubars.append({
            'text': directory.lower(), # remove 'docs/'
            'children': get_menubar_recursively(f"{DOCS}{directory}")
        })
        sidebars[f'/{directory.lower()}/'] = \
            get_menubar_recursively(f"{DOCS}{directory}")

    with open('menubars.yml', 'w', encoding='utf-8') as f:
        yaml.dump(menubars, f, default_flow_style=False)
    
    with open('sidebars.yml', 'w', encoding='utf-8') as f:
        yaml.dump(sidebars, f, default_flow_style=False)


