Introduction
==============================
SOINN(Self-Organizing Incremental Neural Network) implemented by Python.
This project is fork from soinn-python (https://github.com/nkmry/soinn_python).
Thanks for Nakamura-san!

##### Differences from Original project.
* Enable to save/load learned model by joblib.
* Performance Improvement (about 50% faster).
* Appended MNIST clastering Example.
* Corresponding to Python 3.4

![SOINN](https://github.com/fukatani/soinn/blob/master/example.png)

Software Requirements
==============================
* Python (3.4 or later)
* numpy, scipy, matplotlib, scikit-learn, joblib

Installation
==============================

```
git clone https://github.com/fukatani/soinn.git
```


Usage
==============================
Try MNIST exmple,
```
python train_mnist.py
```

Execute test, 
```
python test_soinn.py
```

License
==============================

MIT License.
(http://opensource.org/licenses/mit-license.php)


Copyright
==============================

Copyright (C) 2016, Ryosuke Fukatani

Copyright (C) 2016, Yoshihiro Nakamura
(Original soinn-python)
