# Trying it out


This can run on: [![Binder](https://mybinder.org/badge_logo.svg)](https://gesis.mybinder.org/binder/v2/gh/ianhi/AC295-final-project-JWI/36c01aa3555e690e479fa84354205e268a78ef95?urlpath=lab)


I made separate directories for notebooks and python files becuase otherwise things become a real mess. To import something from a python file in the lib directory put the following at the top of your python notebook:
```python
# set up path for relative imports
import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
```

Then you can do `from lib.____ import ____`


## Dependencies
For jupyter notebook interactions you need `nodejs`, `ipympl`, and `jupyterlab-sidecar`

For installing
```bash
conda install -c conda-forge nodejs ipympl -y
pip install sidecar
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
jupyter labextension install @jupyter-widgets/jupyterlab-sidecar jupyter-matplotlib
```
