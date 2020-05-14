# Trying it out

**Binder**  
Binder is a website that will setup an computational environment for you on their serves: [![Binder](https://mybinder.org/badge_logo.svg)](https://gesis.mybinder.org/binder/v2/gh/ianhi/AC295-final-project-JWI/36c01aa3555e690e479fa84354205e268a78ef95?urlpath=lab)

Unfortunately it's a bit weak of an environment for this application so I'd recommend using docker instead:

**docker**
```
sudo docker run -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes ianhuntisaak/ac295-final-project:v2
```

## Import structure
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


# updating docker image

Follow: https://ropenscilabs.github.io/r-docker-tutorial/04-Dockerhub.html


**build**
```bash
sudo docker build -t ianhuntisaak/ac295-final-project:<tag> .
```

**Verify that it works**
If you use a port other than 8888, e.g. `-p 8889:8888` then you need to change the port in the URL printed in the terminal, can't just copy paste.
```bash
sudo docker run -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes ianhuntisaak/ac295-final-project:<tag>
```

**push to dockerhub**
```bash
sudo docker push ianhuntisaak/ac295-final-project
```

**update the tag version in this readme**
Turns out using the `latest` tag is a bit of a nightmare: https://medium.com/@mccode/the-misunderstood-docker-tag-latest-af3babfd6375
