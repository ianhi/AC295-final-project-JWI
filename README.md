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
