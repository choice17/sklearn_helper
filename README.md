## SKLEARN HELPER

Sklearn_helper repo is a helper library to export sklearn inference model  

Currently,  this library supports  
* [decision tree](#decision_tree)  
* [svm classifier](#svm_classifier)  

## decision_tree  

Usage  

1. get the decision tree function print on console  
```python  
clf.fit(train_X, train_y)
exportTreeFloatC(clf, feature_names) # print to console
```
```
int tree(float a, ...)
{
    if (a <= xx.xx) {
        ...
    }
}
```

2. use it directly in your c code.  

## svm_classifier  

Usage  

1. export model to binary   

```python 
clf = svm.SVC(kernel='rbf', gamma = 0.05, decision_function_shape='ovo' ) #select the algorithm
clf.fit(train_X, train_y)
outputfile = "model.bin"
exportSvmModel(clf, train_X, outputfile)
```

2. load c model (tested in python or use the .so in your c code directly)  

```python
from ctypes import *
import numpy as np

def run():
    lib = CDLL("svm_inference.so", RTLD_GLOBAL)
    model_file = b'model.bin'

    c_float_p = POINTER(c_float)

    svm_load = lib.svm_load
    svm_load.argtypes = [c_char_p]
    svm_load.restype = c_void_p

    svm_free = lib.svm_free
    svm_free.argtypes = [c_void_p]

    svm_pred = lib.svm_pred
    svm_pred.argtypes = [c_void_p, c_float_p]
    svm_pred.restype = c_int

    svm_pred_ext = lib.svm_pred_ext
    svm_pred_ext.argtypes = [c_void_p, c_float_p, c_float_p]
    svm_pred.restype = c_int

    feat_np = np.array([6.6, 2.9, 4.6, 1.3],dtype=np.float32)
    feat_p = (c_float* 4)(*feat_np) #cast to 4 c_float
    feat = cast(feat_p, c_float_p) #cast to c_float_p
    
    prob_p = (c_float*6)() #create empty 6 c_float
    prob = cast(prob_p, c_float_p) #cast to c_float_p

    svm = svm_load(model_file)
    res_cls = svm_pred_ext(svm, feat, prob)
    print("prob: %d %.2f %.2f %.2f %.2f" % (res_cls, prob[0], prob[1], prob[2], prob[3]))

    res_cls = svm_pred(svm, feat)
    print("pred: %d" % (res_cls))

run()
```


