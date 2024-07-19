#!/usr/bin/env python3
from glob import glob
import importlib
import os
import ast
import sys

import configargparse


def to_module(path_name):
    ret = path_name.replace(".py", "").replace("/", ".")
    if ret.endswith("."):
        return ret[:-1]
    return ret


def top_level_functions(body):
    return (f for f in body
        if isinstance(f, ast.FunctionDef)
        and not f.name.startswith("_")
    )


def top_level_classes(body):
    return (f for f in body if isinstance(f, ast.ClassDef))


def parse_ast(filename):
    with open(filename, "rt") as file:
        return ast.parse(file.read(), filename=filename)


def gen_func_rst(func_name, writer):
    writer.write(f"""
{func_name}
{"=" * len(func_name)}

.. autofunction:: {func_name}
""")


def gen_class_rst(class_name, writer):
    writer.write(f"""
{class_name}
{"=" * len(class_name)}

.. autoclass:: {class_name}
    :members:
    :undoc-members:
    :show-inheritance:
""")



# parser
parser = configargparse.ArgumentParser(
    description="generate RST files from <root> module recursively into <dst>/_gen",
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--root", type=str, help="root module to generate docs"
)
parser.add_argument("--dst", type=str, help="destination path to generate RSTs")
parser.add_argument("--exclude", nargs="*", default=[], help="exclude module name")
args = parser.parse_args()
print(args)


gendir = args.dst
os.makedirs(gendir, exist_ok=True)

with open(gendir + f"/{args.root}_package.md", "w") as f_module:
    f_module.write(f"""# {args.root} Package

Documents for {args.root} package.

""")
    os.makedirs(f"{gendir}/{args.root}", exist_ok=True)
    for p in glob(args.root + "/**", recursive=True):
        if p in args.exclude:
            continue
        if "__init__" in p:
            continue
        if not p.endswith(".py"):
            continue
        
        module_name = to_module(p)

        # 1 get functions
        for func in top_level_functions(parse_ast(p).body):
            function_name = func.name
            f_module.write(f" - [{module_name}.{function_name}](./{args.root}/{module_name}.{function_name}.md)\n")
            print(f"[INFO] generating {func.name} in {module_name}")
            # 1.2 generate RST
            with open(f"{gendir}/{args.root}/{module_name}.{function_name}.rst", "w") as f_rst:
                gen_func_rst(f"{module_name}.{function_name}", f_rst)
        
        # 2 get classes
        for clz in top_level_classes(parse_ast(p).body):
            class_name = clz.name
            f_module.write(f" - [{module_name}.{class_name}](./{args.root}/{module_name}.{class_name}.md)\n")
            print(f"[INFO] generating {clz.name} in {module_name}")
            # 1.2 generate RST
            with open(f"{gendir}/{args.root}/{module_name}.{class_name}.rst", "w") as f_rst:
                gen_class_rst(f"{module_name}.{class_name}", f_rst)

