# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'hhg9'
copyright = '2025, Ben Griffin'
author = 'Ben Griffin'
release = '0.3.0a0'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',        # Markdown source, to match the rest of docs/
    'sphinx_design',      # dropdowns, cards, tab-sets
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

myst_enable_extensions = [
    'colon_fence',        # ::: fences, easier to read than ```{directive}
    'deflist',
    'attrs_inline',
    'substitution',
]

# Marginalia are asides: never let one silently become a heading target.
myst_heading_anchors = 3

# Tutorial chapters are jupytext-paired MyST notebooks: their executable cells
# are ```{code-cell} directives, which plain myst_parser does not know. Render
# them as ordinary python code blocks; swapping myst_parser for myst_nb later
# (which executes them during the build) replaces this shim without touching
# the chapters. Verification lives in docs/tutorial/verify.sh meanwhile.
from sphinx.directives.code import CodeBlock


class _CodeCell(CodeBlock):
    required_arguments = 0
    optional_arguments = 1

    def run(self):
        self.arguments = ['python']
        return super().run()


def setup(app):
    app.add_directive('code-cell', _CodeCell)

templates_path = ['_templates']
# docs/ also holds the paper drafts and working notes as Markdown. MyST would
# otherwise pick every one of them up as an orphan page, so the docs build is
# opt-in: only index.rst and tutorial/ are part of it for now.
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    'arxiv', 'paper_figures', 'dggs', 'supply_line',
    'paper*.md', 'dggs*.md', 'release-notes*.md',
    'glossary.md', 'enumeration.md',
]

source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    'repository_url': 'https://github.com/MrBenGriffin/hex9',
    'use_repository_button': True,
    'use_issues_button': True,
    'show_navbar_depth': 2,
    'show_toc_level': 2,
}
