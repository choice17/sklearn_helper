"""
sklearn_helper
1. export multiclass svm kernel / linear to svm_inference.c
2. export decision tree
"""

from sklearn.tree import _tree
from sklearn import svm

# export tree to python https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
def exportTreeFloatC(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("int tree({})".format(", ".join(["double " + f for f in feature_names])))
    print("{")
    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if ({} <= {:.6f}) {{".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}}} else {{ // if {} > {:.6f}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
            print("{}}}".format(indent))
        else:
            print ("{}return {};".format(indent, np.argmax(tree_.value[node])))
    recurse(0, 1)
    print("}")


def exportTreeIntC(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("int tree({})".format(", ".join(["int " + f for f in feature_names])))
    print("{")
    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            threshold = int(np.floor(threshold))
            print("{}if ({} <= {:d}) {{".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}}} else {{ // if {} > {:d}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
            print("{}}}".format(indent))
        else:
            print ("{}return {};".format(indent, np.argmax(tree_.value[node])))
    recurse(0, 1)
    print("}")


def determineGamma(params, n_cls, n_feat, trainX):
    p = params
    g = -1.0
    if p['kernel'] == 'linear':
        return g
    elif p['kernel'] == 'rbf':
        if type(p['gamma']) == str:
            if p['gamma'].find('auto')>-1:
                g = 1 / n_feat
            else:
                g = 1 / (n_feat * trainX.var())
        else:
            g = p['gamma']
    return g  
        
def exportSvmModel(svm_model, trainX, outputfile="model.bin", verbose=0):
    """
    /**
     *@brief Dump svm mobel to output file which can be recognized by helper c library
     *@param[in] svm_model   Sklearn svm model (multiclass/ kernel(rbf) or linear)
     *@param[in] trainX      nunmpy 2d array for var()
     *@see also ""
    **/

    /* svm_inference.h */
    typedef struct {
        int version[3];
        char contact[32];
        char description[16];
        char kernel[16]; /**< Kernel name ex. rbf/linear */
        int n_cls; /**< Number of class */
        int n_feat; /**< Number of feature of the SVM */
        int nSV; /**< Number of Support vectors */
    } SVM_HEADER;

    typedef struct {
        int *nv; /**< Number of support vectors for each class Dimension [n_cls,1] */
        float gamma; /**< Gamma value of rbf kernel */
        float *a; /**< dual coef Dimension [n_cls-1, nSV] */
        float *b; /**< bias Dimension [(n_cls * (n_cls-1))/2,1] */
        float *sv; /**< Support vectors dimension [nSV, n_feat] */
    } SVM_DATA;

    typedef struct {
        SVM_HEADER header;
        SVM_DATA data;
    } SVM_MODEL;
    
    SVM_MODEL *svm_load(const char *model_file);
    int svm_pred(const SVM_MODEL *svm, const float *feat);
    int svm_pred_ext(const SVM_MODEL *svm, const float *feat, float *prob);
    svm_free(SVM_MODEL *svm);
    """
    file = outputfile
    _version_major = 0
    _version_minor = 0
    _version_rc = 1
    _contact = 'tcyu@umich.edu'
    _contact = _contact + "\0" * (32 - len(_contact)) 
    _description = 'sklearn_svm'
    _description = _description + "\0" * (16 - len(_description))
    m = svm_model
    p = m.get_params()
    
    _kernel = p['kernel']
    _kernel = _kernel + "\0" * (16 - len(_kernel))
    _sv = m.support_vectors_
    _nv = m.n_support_
    _a  = m.dual_coef_
    _b  = m._intercept_
    _nSV, n_feat = _sv.shape
    n_cls = len(_nv)
    _g = determineGamma(p, n_cls, n_feat, trainX)    
    int_bin = [_version_major, _version_minor,_version_rc]
    str_bin = [_contact, _description, _kernel]
    model_int = [n_cls, n_feat, _nSV, _nv]
    float_bin = [_g,_a,_b,_sv]
    
    with open(file, 'wb') as f:
        for i in int_bin:
            f.write(np.array(i,dtype=int).tostring())
        for i in str_bin:
            f.write(i.encode())
        for i in model_int:
            f.write(np.array(i,dtype=np.int32).tostring())
        for i in float_bin:
            f.write(np.array(i,dtype=np.float32).tostring())
    if verbose:
        print("SVM Header Info")
        print("Version:%d.%d.%d" % (_version_major, _version_minor, _version_rc))
        print("Contact:%s" % (_contact))
        print("Description:%s" % (_description))
        print("SVM Kernel:%s" % (_kernel))
        print("Number of classes:%d" % (n_cls))
        print("Number of features:%d" % (n_feat))
        print("Number of support vectors:%d" % (_nSV))
        print("nv:[%d, %d, ...]" % (_nv[0], _nv[1]))
        print("g:%.4f" % (_g))
        print("a:[%.4f, %.4f, %.4f, ...]" % (_a[0,0], _a[0,1], _a[0,2]))
        print("b:[%.4f, %.4f, %.4f, ...]" % (_b[0], _b[1], _b[2]))
        print("sv:[%.4f, %.4f, %.4f, ...]" % (_sv[0,0], _sv[0,1], _sv[0,2]))