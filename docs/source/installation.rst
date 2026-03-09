Installation
=======================================

Install Python
-----------------
ScenicRules requires Python 3.10 or higher. If you don't have a compatible Python version, you can download it from the `official Python website <https://www.python.org/downloads/>`_ or use `pyenv <https://github.com/pyenv/pyenv>`_ to install it as follows.

.. code-block:: bash

   pyenv install 3.10.18

Set up a Python Virtual Environment
---------------------------------------
We recommend installing ScenicRules within an isolated Python virtual environment to prevent dependency conflicts. We provide two options for setting up the environment:

Option 1: Using venv
^^^^^^^^^^^^^^^^^^^^^^^^

You can create and activate a virtual environment using Python's built-in `venv <https://docs.python.org/3/library/venv.html>`_ module as follows.
    
.. code-block:: bash

    # Create a virtual environment named "venv_scenicrules"; you only need to do this once
    python3 -m venv venv_scenicrules
    # Activate the virtual environment; you need to do this every time you start a new terminal session
    source venv_scenicrules/bin/activate

Option 2: Using pyenv+virtualenv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Alternatively, you can use `pyenv <https://github.com/pyenv/pyenv>`_ along with `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_ to manage your Python versions and virtual environments. You can create and activate a virtual environment as follows.
    
.. code-block:: bash

    # Create a virtual environment named "venv_scenicrules" with Python 3.10.18; you only need to do this once
    pyenv virtualenv 3.10.18 venv_scenicrules
    # Activate the virtual environment; you need to do this every time you start a new terminal session
    pyenv activate venv_scenicrules

Install ScenicRules
-----------------------
Once you have set up and activated your Python virtual environment, you can install ScenicRules and its dependencies from the repository as follows.

.. code-block:: bash

   git clone https://github.com/BerkeleyLearnVerify/ScenicRules.git
   cd ScenicRules
   python -m pip install -e .

The above commands will install ScenicRules in "editable" mode, which allows you to make changes to the code and have them reflected without needing to reinstall the package.
