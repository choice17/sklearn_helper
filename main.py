

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
