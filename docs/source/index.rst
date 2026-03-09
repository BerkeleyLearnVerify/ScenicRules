.. ScenicRules documentation master file, created by
   sphinx-quickstart on Wed Feb 18 18:27:01 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ScenicRules!
=======================================

ScenicRules is an autonomous driving benchmark designed to evaluate systems against complex, multi-objective specifications in diverse driving contexts. It integrates the following key features:

- Multi-Objective Specification: Supports the formalization of conflicting driving objectives and explicit priority relations using the `Rulebook <https://arxiv.org/abs/1902.09355>`_ framework.
- Abstract Scenario Representation: Leverages the `Scenic <https://scenic-lang.org>`_ probabilistic programming language to model driving contexts in an expressive, compact, and interpretable manner.

ScenicRules is designed and implemented by Kevin Kai-Chun Chang, Ekin Beyazit, Alberto Sangiovanni-Vincentelli, Tichakorn Wongpiromsarn, and Sanjit A. Seshia. See `our paper <https://www.arxiv.org/abs/2602.16073>`_ for more details.

Table of Contents
=======================================
.. toctree::
   :maxdepth: 2
   :caption: Getting Started with ScenicRules

   installation
   overview
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: ScenicRules Internals

   preprocessing
   rules
   rulebook
   scenarios

.. toctree::
   :maxdepth: 2
   :caption: ScenicRules as a Benchmark

   falsification
   evaluation_human

Citation
=======================================
If you use ScenicRules in your work, please cite the following paper:

.. code-block:: bibtex

   @inproceedings{chang2026scenicrules,
      title={{ScenicRules}: An Autonomous Driving Benchmark with Multi-Objective Specifications and Abstract Scenarios},
      author={Chang, Kevin Kai{-}Chun and Beyazit, Ekin and Sangiovanni-Vincentelli, Alberto and Wongpiromsarn, Tichakorn and Seshia, Sanjit A.},
      booktitle={IEEE Intelligent Vehicles Symposium (IV)},
      year={2026}
   }

Liscense
=======================================
ScenicRules is licensed under the `3-Clause BSD License <https://opensource.org/license/bsd-3-clause>`_.
