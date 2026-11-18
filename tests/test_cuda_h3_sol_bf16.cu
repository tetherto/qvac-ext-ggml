#include "ggml-cuda/h3-sol-bf16.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<typename T> static T * upload(const std::vector<T> & values) {
    T * pointer; check(cudaMalloc(&pointer,values.size()*sizeof(T)));
    check(cudaMemcpy(pointer,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
    return pointer;
}

static __global__ void correction_kernel(const float * q, const float * kc, const float * vc,
        const int * route, const float2 * exact_meta, const float * exact_values,
        float * correction, float * output, int tokens, int first, int stop, float scale) {
    extern __shared__ __align__(16) unsigned char scratch[];
    h3_sol_accumulate_bf16(reinterpret_cast<const float2 *>(q),64,0,tokens,kc,vc,route,
                           first,stop,scale,correction,scratch);
    for (int row=threadIdx.y;row<64;row+=blockDim.y) if (threadIdx.x==0) {
        float * metadata=correction+row*h3_sol_correction_stride;
        metadata[1]=h3_sol_merge_metadata(exact_meta[row],metadata).y;
    }
    __syncthreads();
    for (int index=threadIdx.y*32+threadIdx.x;index<64*128;index+=128) {
        const int row=index/128,c=index%128;
        const float * metadata=correction+row*h3_sol_correction_stride;
        output[index]=(exact_values[index]*metadata[2]+metadata[4+c]*metadata[3])/metadata[1];
    }
}

static float rounded(float x) { return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)); }
static float half_rounded(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }

static void run_case(int tokens,int first,int stop,float scale,bool keep_all,bool large,bool negative_only=false) {
    const int blocks=(tokens+63)/64;
    std::vector<float> q(64*128),kc(blocks*128),vc(blocks*128),exact_values(64*128);
    std::vector<float2> exact_meta(64);
    std::vector<int> route(blocks);
    for (int row=0;row<64;++row) {
        exact_meta[row]=make_float2(float(row%5)-2.f,.5f+.03f*row);
        if (negative_only) exact_meta[row].x=-200.f;
        for (int c=0;c<128;++c) {
            q[row*128+c]=float((row+3*c)%17-8)/64.f;
            exact_values[row*128+c]=.1f*std::cos(float(row+c));
        }
        if (large) q[row*128]=(!negative_only && row%2 ? -13.f : 13.f)/scale;
    }
    for (int b=0;b<blocks;++b) {
        route[b]=keep_all || b%3==0;
        for (int c=0;c<128;++c) {
            kc[b*128+c]=float((b*7+c*3)%19-9)/128.f;
            vc[b*128+c]=std::min(64,tokens-b*64)*half_rounded(.04f*std::sin(float(b+3*c)));
        }
        if (large) kc[b*128]=negative_only || b%2 ? -8.f : 8.f;
    }
    auto dq=upload(q);auto dk=upload(kc);auto dv=upload(vc);auto dr=upload(route);
    auto dm=upload(exact_meta);auto de=upload(exact_values);
    std::vector<float> zeros(64*h3_sol_correction_stride);
    auto dw=upload(zeros);float * result;check(cudaMalloc(&result,q.size()*sizeof(float)));
    correction_kernel<<<1,dim3(32,4),sizeof(h3_sol_bf16_workspace)>>>(dq,dk,dv,dr,dm,de,dw,result,tokens,first,stop,scale);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    std::vector<float> actual(q.size());check(cudaMemcpy(actual.data(),result,q.size()*sizeof(float),cudaMemcpyDeviceToHost));
    double energy=0.,error=0.;float maximum_error=0.f;
    for (int row=0;row<64;++row) {
        std::vector<double> query(128),numerator(128);
        for (int c=0;c<128;++c) query[c]=rounded(half_rounded(half_rounded(q[row*128+c])*half_rounded(scale)));
        float maximum=-FLT_MAX/2.f;double mass=0.;
        for (int start=first+1;start<stop-1;start+=16) {
            std::vector<double> scores(16);
            bool active=false;
            for (int col=0;col<16;++col) {
                const int b=start+col;
                if (b>=stop-1 || route[b]) continue;
                active=true;
                for (int c=0;c<128;++c) scores[col]+=query[c]*rounded(kc[b*128+c]);
            }
            if (!active) continue;
            float next=maximum;
            for (int col=0;col<16;++col) if (start+col<stop-1 && !route[start+col]) next=std::max(next,float(scores[col])+FATTN_KQ_MAX_OFFSET);
            const double alpha=std::exp(double(maximum-next));
            mass*=alpha;for (double & value:numerator) value*=alpha;
            for (int col=0;col<16;++col) {
                const int b=start+col;
                if (b>=stop-1 || route[b]) continue;
                const double probability=std::exp(double(float(scores[col])-next));
                mass+=probability*std::min(64,tokens-64*b);
                const double quantized=rounded(float(probability));
                for (int c=0;c<128;++c) numerator[c]+=quantized*rounded(vc[b*128+c]);
            }
            maximum=next;
        }
        const float combined=std::max(maximum,exact_meta[row].x);
        const double a=std::exp(double(exact_meta[row].x-combined)),b=std::exp(double(maximum-combined));
        const double denominator=exact_meta[row].y*a+mass*b;
        for (int c=0;c<128;++c) {
            const double expected=(exact_values[row*128+c]*a+numerator[c]*b)/denominator;
            const float value=actual[row*128+c];
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite BF16 correction");
            const double delta=value-expected;
            maximum_error=std::max(maximum_error,float(std::abs(delta)));error+=delta*delta;energy+=expected*expected;
        }
    }
    const double relative=std::sqrt(error/std::max(energy,1e-300));
    if (maximum_error>1e-4 || relative>(large ? 1e-3 : 1e-4)) throw std::runtime_error("BF16 Tensor Core correction differs from independent quantized-input double oracle");
    void * allocations[]={dq,dk,dv,dr,dm,de,dw,result};for (void * pointer:allocations) check(cudaFree(pointer));
    std::printf("PASS BF16/FP32 tokens=%d first=%d stop=%d all_keep=%d large=%d negative_only=%d max_abs=%.9g relative_rms=%.9g\n",
                tokens,first,stop,keep_all,large,negative_only,maximum_error,relative);
}

int main() {
    try {
        int count;check(cudaGetDeviceCount(&count));if (count!=1) return 77;
        cudaDeviceProp properties;check(cudaGetDeviceProperties(&properties,0));if (properties.major<8) return 77;
        for (bool large:{false,true}) {
            run_case(257,0,5,1.f,false,large);
            run_case(1025,3,11,128.f/std::sqrt(128.f),false,large);
            run_case(2051,0,33,.125f,false,large);
            run_case(1025,0,17,.125f,true,large);
        }
        run_case(1025,0,17,1.f,false,true,true);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL %s\n",error.what());return 1;
    }
}
