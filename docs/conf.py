# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NeuralAmpModelerCore'
copyright = '2023-present Steven Atkinson'
author = 'Neural Amp Modeler Contributors'
release = '0.4.0'
version = '0.4.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'breathe',
    'sphinxcontrib.mermaid',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']  # Commented out until _static directory is created

# -- Breathe configuration ----------------------------------------------------
# https://breathe.readthedocs.io/

breathe_projects = {
    'NeuralAmpModelerCore': 'doxygen/xml',
}
breathe_default_project = 'NeuralAmpModelerCore'
breathe_default_members = ('members', 'undoc-members')

# -- Intersphinx configuration -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    'cpp': ('https://en.cppreference.com/mwiki/', None),
}

# -- Extension configuration --------------------------------------------------

# Autodoc settings
autodoc_mock_imports = ['Eigen', 'nlohmann']
