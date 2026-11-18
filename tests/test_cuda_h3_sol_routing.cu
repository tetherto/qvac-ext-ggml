#include "ggml-cuda/fattn-common.cuh"
#include "ggml-cuda/h3-sol-routing.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

static bool eligible(int qb, int kb, const ggml_cuda_h3::sol_metadata & meta) {
    if (qb*64<meta.video_start || (qb+1)*64>meta.video_end || kb*64<meta.video_start || (kb+1)*64>meta.video_end) return false;
    const int q0=(qb*64-meta.video_start)/meta.frame_tokens,q1=((qb+1)*64-1-meta.video_start)/meta.frame_tokens;
    const int k0=(kb*64-meta.video_start)/meta.frame_tokens,k1=((kb+1)*64-1-meta.video_start)/meta.frame_tokens;
    return !(k0<=q1+1 && k1>=q0-1);
}

static void run_case(bool tiny_query, int mode, float beta = -2.f, bool use_gemm = false) {
    constexpr int blocks=17,heads=56,channels=128;
    const int elements=heads*blocks*channels;
    std::vector<float> q(elements),k(elements);
    for (int h=0;h<heads;++h) for (int b=0;b<blocks;++b) for (int c=0;c<channels;++c) {
        const int i=(h*blocks+b)*channels+c;
        q[i]=tiny_query ? (c==0 ? 1e-6f : 0.f) : .4f*std::sin(float(h+3*b+7*c));
        k[i]=tiny_query ? (b==12 && c==0 ? -1.f : 0.f) : .2f*std::cos(float(2*h+11*b+5*c));
    }
    float * dq,*dk,*dm,*dv,*ds=nullptr;int *dr;
    cublasHandle_t handle=nullptr;
    check(cudaMalloc(&dq,elements*sizeof(float)));check(cudaMalloc(&dk,elements*sizeof(float)));
    check(cudaMalloc(&dm,heads*channels*sizeof(float)));check(cudaMalloc(&dv,heads*channels*sizeof(float)));
    check(cudaMalloc(&dr,(heads*blocks*blocks+4)*sizeof(int)));
    check(cudaMemcpy(dq,q.data(),elements*sizeof(float),cudaMemcpyHostToDevice));check(cudaMemcpy(dk,k.data(),elements*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemset(dr,0,(heads*blocks*blocks+4)*sizeof(int)));
    h3_sol_kc_stats<<<heads,channels>>>(dk,dm,dv,blocks);
    ggml_cuda_h3::sol_metadata meta{ggml_cuda_h3::sol_magic,3,mode,65,961,128,beta};
    if (use_gemm) {
        check(cudaMalloc(&ds, heads*blocks*blocks*sizeof(float)));
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        if (!h3_sol_proxy_scores(handle, nullptr, dq, dk, ds, blocks)) throw std::runtime_error("Proxy GEMM failed");
        cublasPointerMode_t restored; CUBLAS_CHECK(cublasGetPointerMode(handle, &restored));
        if (restored != CUBLAS_POINTER_MODE_DEVICE) throw std::runtime_error("Proxy GEMM changed caller pointer mode");
    }
    h3_sol_routes<<<dim3(blocks,heads),256>>>(dq,dk,dr,blocks,1.f,dm,dv,meta,ds);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    std::vector<int> actual(heads*blocks*blocks+4);check(cudaMemcpy(actual.data(),dr,actual.size()*sizeof(int),cudaMemcpyDeviceToHost));
    int pruned=0,ambiguous=0,unfloored_pruned=0;
    for (int h=0;h<heads;++h) for (int qb=0;qb<blocks;++qb) {
        std::vector<double> scores(blocks);double mean=0.,variance=0.;
        for (int kb=0;kb<blocks;++kb) for (int c=0;c<channels;++c) scores[kb]+=double(q[(h*blocks+qb)*channels+c])*k[(h*blocks+kb)*channels+c];
        {
            for (int c=0;c<channels;++c) {
                double mu=0.,second=0.;
                for (int kb=0;kb<blocks;++kb) {const double x=k[(h*blocks+kb)*channels+c];mu+=x;second+=x*x;}
                mu/=blocks;const double query=q[(h*blocks+qb)*channels+c];
                mean+=query*mu;variance+=query*query*std::max(second/blocks-mu*mu,0.);
            }
            const float log2e=1.4426950408889634f;variance+=1e-6f/(log2e*log2e);
        }
        const double threshold=mean+meta.beta*std::sqrt(variance);
        for (int kb=0;kb<blocks;++kb) {
            const bool protected_block=!eligible(qb,kb,meta);
            const bool expected=mode==7 || protected_block || !(scores[kb]<threshold);
            const float log2e=1.4426950408889634f;
            const double without_floor=mean+meta.beta*std::sqrt(std::max(0.,variance-1e-6f/(log2e*log2e)));
            if (!protected_block && scores[kb]<without_floor) ++unfloored_pruned;
            const bool keep=actual[(h*blocks+qb)*blocks+kb]!=0;
            if (!protected_block && std::abs(scores[kb]-threshold)<1e-8*std::max(1.,std::abs(threshold))) {++ambiguous;continue;}
            if (keep!=expected) throw std::runtime_error("GPU routing disagrees with independent double-precision selector");
            if (!keep) ++pruned;
        }
    }
    if (!tiny_query && mode==6 && beta>=0.f && !pruned) throw std::runtime_error("Active diagonal routing not exercised");
    if (tiny_query && pruned) throw std::runtime_error("NVIDIA variance floor failed to protect a near-zero query centroid");
    if (tiny_query && mode==6 && !unfloored_pruned) throw std::runtime_error("Variance-floor CPU negative control not exercised");
    if (handle) CUBLAS_CHECK(cublasDestroy(handle));
    if (ds) check(cudaFree(ds));
    void * allocations[]={dq,dk,dm,dv,dr};for (void * p:allocations) check(cudaFree(p));
    std::printf("PASS tiny_query=%d mode=%d beta=%.4g gemm=%d pruned=%d ambiguous_near_threshold=%d\n",tiny_query,mode,beta,use_gemm,pruned,ambiguous);
}

int main() {
    try {
        int count;check(cudaGetDeviceCount(&count));if (count!=1) return 77;
        for (bool gemm:{false,true}) for (bool tiny:{false,true}) for (int mode:{6,7}) run_case(tiny,mode,-2.f,gemm);
        for (bool gemm:{false,true}) for (int mode:{6,7}) for (float beta:{0.f,1.f}) run_case(false,mode,beta,gemm);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL %s\n",error.what());return 1;
    }
}
