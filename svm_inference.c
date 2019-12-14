#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "svm_inference.h"

//#define SVM_DEBUG

#define FMT(fmt) "[SVM] " fmt
#define INFO(fmt, args...) printf(FMT(fmt), ##args)
#ifdef SVM_DEBUG
#define DEBUG(fmt, args...) INFO(fmt, ##args)

static void printSvmData(const SVM_DATA *data);

#else
#define DEBUG(fmt, args...)
#endif /* SVM_DEBUG */

#define KERNEL_LINEAR "linear"
#define KERNEL_RBF "rbf"

static float linear(const float *feat, const float *sv, int n, const void *args);
static float rbf(const float *feat, const float *sv, int n, const void *args);
//static float *kernel(float *feat, float *sv, void *args);
static inline float dot(const float *x, const float *y, int n);
static void printSvmHeaderInfo(const SVM_HEADER *h);
static int initSvmData(SVM_MODEL *svm);
static void readSvmData(FILE *fp, SVM_MODEL *svm, int pos);
static int freeSvmData(SVM_DATA *data);

static inline float dot(const float *x, const float *y, int n)
{
    int i;
    float r = 0.0;
    for (i = 0; i < n; i++) {
        r += x[i] * y[i];
    }
    return r;
}

static float linear(const float *feat, const float *sv, int n, const void *args)
{
    return dot(feat, sv, n);
}

static float rbf(const float *feat, const float *sv, int n, const void *args)
{
    float r=0.0, subt, _s, g;
    int i;
    g = *(float *)args;
    for (i = 0; i < n; i++) {
        subt = feat[i] - sv[i];
        _s = subt * subt;
        r += _s;
    }
    return exp(-g * r);
}

static void printSvmHeaderInfo(const SVM_HEADER *h)
{
    INFO("SVM Header Info.%s\n", "");
    INFO("Version:%d.%d.%d\n", h->version[0], h->version[1], h->version[2]);
    INFO("Contact:%s\n", h->contact);
    INFO("Description:%s\n", h->description);
    INFO("SVM Kernel:%s\n", h->kernel);
    INFO("Number of classes:%d\n", h->n_cls);
    INFO("Number of features:%d\n", h->n_feat);
    INFO("Number of support vectors:%d\n", h->nSV);
    if ((h->version[0] != VERSION_MAJOR) ||
        (h->version[1] != VERSION_MINOR) ||
        (h->version[2] != VERSION_RC))
        INFO("Warning! Model version(v%d.%d.%d) is not consistent to this library(v%d.%d.%d).\n",
            h->version[0], h->version[1], h->version[2],
            VERSION_MAJOR, VERSION_MINOR, VERSION_RC);
}

static int initSvmData(SVM_MODEL *svm)
{
    svm->data.nv = malloc(sizeof(int) * svm->header.n_cls);
    if (svm->data.nv == NULL) {
        INFO("Cannot allocate memory for svm number of support vector for each classes.\n");
        return -1;
    }
    svm->data.a = malloc(sizeof(float) * (svm->header.n_cls-1) * (svm->header.nSV));
    if (svm->data.a == NULL) {
        INFO("Cannot allocate memory for svm dual coefficients.\n");
        return -1;
    }
    svm->data.b = malloc(sizeof(float) * (((svm->header.n_cls) * (svm->header.n_cls-1)) / 2) );
    if (svm->data.b == NULL) {
        INFO("Cannot allocate memory for svm bias.\n");
        return -1;
    }
    svm->data.sv = malloc(sizeof(float) * (svm->header.nSV) * (svm->header.n_feat) );
    if (svm->data.sv == NULL) {
        INFO("Cannot allocate memory for svm support vectors.\n");
        return -1;
    }
    return 0;
}

#ifdef SVM_DEBUG
static void printSvmData(const SVM_DATA *data)
{
    INFO("nv:[%d, %d, ...]\n", data->nv[0], data->nv[1]);
    INFO("g:%.4f\n", data->gamma);
    INFO("a:[%.4f, %.4f, %.4f, ...]\n", data->a[0], data->a[1], data->a[2]);
    INFO("b:[%.4f, %.4f, %.4f, ...]\n", data->b[0], data->b[1], data->b[2]);
    INFO("sv:[%.4f, %.4f, %.4f, ...]\n", data->sv[0], data->sv[1], data->sv[2]);
}
#endif /* SVM_DEBUG */

static void readSvmData(FILE *fp, SVM_MODEL *svm, int pos)
{
    int size_nv = sizeof(int) * svm->header.n_cls;
    int size_g = sizeof(int);
    int size_a = sizeof(float) * (svm->header.n_cls-1) * (svm->header.nSV);
    int size_b = sizeof(float) * (((svm->header.n_cls) * (svm->header.n_cls-1)) / 2);
    int size_sv = sizeof(float) * (svm->header.nSV) * (svm->header.n_feat);
    int size_total = 0;

    SVM_DATA *data = &svm->data;
    pos += size_nv;
    fread(data->nv, 1, size_nv, fp);
    fseek(fp, pos, SEEK_SET);
    pos += size_g;
    fread(&data->gamma, 1, size_g, fp);
    fseek(fp, pos, SEEK_SET);
    pos += size_a;
    fread(data->a, 1, size_a, fp);
    fseek(fp, pos, SEEK_SET);
    pos += size_b;
    fread(data->b, 1, size_b, fp);
    fseek(fp, pos, SEEK_SET);
    pos += size_sv;
    fread(data->sv, 1, size_sv, fp);
    fseek(fp, 0, SEEK_END);
    size_total = ftell(fp);
    if (size_total != (pos)) {
        INFO("Read file size not correct(expected:%d vs read:%d)!\n", size_total, pos);
    }
#ifdef SVM_DEBUG
    printSvmData(data);
#endif /* !SVM_DEBUG */
    return;

}

static int freeSvmData(SVM_DATA *data)
{
    free(data->nv);
    free(data->a);
    free(data->b);
    free(data->sv);
    return 0;
}

SVM_MODEL *svm_load(const char *model_file)
{
    int pos = 0;
    FILE *fp = fopen(model_file, "rb");
    if (fp == NULL) {
        INFO("Cannot read requested file %s\n", model_file);
        return NULL;
    }
    SVM_MODEL *svm = malloc(sizeof(SVM_MODEL));
    pos = sizeof(SVM_HEADER);
    fread(&svm->header, 1, pos, fp);
    printSvmHeaderInfo(&svm->header);
    fseek(fp, pos, SEEK_SET);
    if (initSvmData(svm)) {
        INFO("Cannot init svm data\n");
        return NULL;
    }
    readSvmData(fp, svm, pos);
    return svm;
}

int svm_pred_ext(const SVM_MODEL *svm, const float *feat, float *prob)
{
    float (*K)(const float*, const float*, int, const void*) = NULL;
    int *start;
    float *kvalue;
    float *retval = prob;
    int *vote;
    int idx = 0, P = 0, cls = -1;
    int si, sj, ci, cj, i, j, k;
    float _sum, *c1, *c2;

    const SVM_HEADER *h = &svm->header;
    const SVM_DATA *d = &svm->data;

    /* Determine kernel type */
    if (strcmp(h->kernel, KERNEL_RBF) == 0) {
        K = &rbf;
    } else if (strcmp(h->kernel, KERNEL_LINEAR) == 0) {
        K = &linear;
    } else {
        INFO("Not supported kernel type(%s)\n.", h->kernel);
        return -1;
    }

    /* Define the start and end index for support vectors for each class */
    start = malloc(sizeof(int) * h->n_cls);
    kvalue = malloc(sizeof(float) * h->nSV);
    vote = malloc(sizeof(int) * h->n_cls);
    memset(vote, 0, sizeof(int) * h->n_cls);

    for (i = 0; i < h->nSV; ++i) {
        idx = i * h->n_feat;
        kvalue[i] = K(feat, &d->sv[idx], h->n_feat, (void *)&d->gamma);
        DEBUG("kvalue[%d](%.3f) = k(a, sv[%d](%.3f, %.3f, %.3f, %.3f))\n", i,
            kvalue[i], idx, d->sv[idx], d->sv[idx+1], d->sv[idx+2],d->sv[idx+3]);
    }

    start[0] = 0;
    for (i = 1; i < h->n_cls; ++i) {
        start[i] = start[i-1] + d->nv[i-1];
    }

    for (i = 0; i < h->n_cls; ++i) {
        for (j = i+1; j < h->n_cls; ++j) {
            _sum = 0;
            si = start[i];
            sj = start[j];
            ci = d->nv[i];
            cj = d->nv[j];
            c1 = &d->a[(j-1) * h->nSV];
            c2 = &d->a[(i) * h->nSV];

            for (k = 0; k < ci; ++k) {
                DEBUG("retval[%d] += a[%d,%d](%.3f) * kvalue[%d](%.3f), ci:%d\n",
                P, j-1, si+k, c1[si+k], si+k, kvalue[si+k], ci);
                _sum += c1[si+k] * kvalue[si+k];
            }
            for (k = 0; k < cj; ++k) {
                DEBUG("retval[%d] += a[%d,%d](%.3f) * kvalue[%d](%.3f), cj:%d\n",
                P, i, sj+k, c2[sj+k], sj+k, kvalue[sj+k], cj);
                _sum += c2[sj+k] * kvalue[sj+k];
            }
            DEBUG("ret[%d] = %.4f => %.4f\n", P, _sum, _sum + d->b[P]);
            _sum += d->b[P];
            retval[P] = _sum;

            if(retval[P] > 0) {
                ++vote[i];
            } else {
                ++vote[j];
            }
            DEBUG("vote[%d %d %d]\n", vote[0], vote[1], vote[2]);
            P++;
        }
    }
    cls = 0;
    for (i=1 ; i < h->n_cls; i++) {
        DEBUG("vote[%d]:%d vote[cls](%d) cls:%d\n", i, vote[i], vote[cls], cls);
       if (vote[i] > vote[cls]) {
           cls = i;
       }
    }

    free(start);
    free(kvalue);
    free(vote);

    return cls;
}

int svm_pred(const SVM_MODEL *svm, const float *feat)
{
    int cls = 0;
    int nr_cls = ((svm->header.n_cls)*(svm->header.n_cls-1))/2;

    float *retval = malloc(sizeof(float)*nr_cls);

    cls = svm_pred_ext(svm, feat, retval);
    
    free(retval);
    
    return cls;
}


int svm_free(SVM_MODEL *svm)
{
    freeSvmData(&svm->data);
    free(svm);
    return 0;
}

int main(void)
{
    const char *file = "model.bin";
    SVM_MODEL *svm = svm_load(file);
    float feat[4] = {6.6, 2.9, 4.6, 1.3};
    float *prob = malloc(sizeof(float) * ((svm->header.n_cls)*(svm->header.n_cls-1))/2);
    int cls = svm_pred_ext(svm, feat, prob);
    INFO("input: %.2f, %.2f, %.2f ,%.2f\n",
        feat[0], feat[1], feat[2], feat[3]);
    INFO("cls:%d\n", cls);
    INFO("prob: %.2f, %.2f, %.2f\n",
        prob[0], prob[1], prob[2]);
    svm_free(svm);
    return 0;
}