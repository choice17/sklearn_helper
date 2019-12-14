#ifndef SVM_INFERENCE_H_
#define SVM_INFERENCE_H_

#define VERSION_MAJOR 0
#define VERSION_MINOR 0
#define VERSION_RC 1

/* svm_inference.h */
typedef struct {
    int version[3];
    char contact[32];
    char description[16];
    char kernel[16]; /**< Kernel name ex. rbf/linear */
    int n_cls; /**< Number of class */
    int n_feat; /**< Number of feature of the SVM */
    int nSV; /**< Number of support vectors */
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
int vm_free(SVM_MODEL *svm);

#endif /* !SVM_INFERENCE_H_ */
