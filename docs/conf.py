import subprocess

branch = subprocess.check_output(["/bin/bash", "-c", "git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || git rev-parse HEAD"]).strip().decode()
project = 'Triton-Ascend Guidebook'
author = 'Ascend'
release = '1.0.0'

extensions = [
    'myst_parser', 
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

language = 'zh_CN'

templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}