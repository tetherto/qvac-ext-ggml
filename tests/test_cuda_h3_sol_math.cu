#include "ggml-cuda/fattn-common.cuh"
#include "test_cuda_h3_sol_reference.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <typename T> static T * upload(const std::vector<T> & values) {
    T * result;
    check(cudaMalloc(&result, values.size()*sizeof(T)));
    check(cudaMemcpy(result, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice));
    return result;
}

static __global__ void correction_kernel(const float * q, const float * kc, const float * vc,
        const int * routes, const float2 * exact_meta, const float * exact_values,
        float * workspace, float * output, int tokens, int first, int stop, float scale,
        bool exact_terms, const half * raw_k, const half * raw_v) {
    if (exact_terms) h3_sol_accumulate<true>(reinterpret_cast<const float2 *>(q), 64, 0, tokens,
                      kc, vc, routes, first, stop, scale, workspace,raw_k,128,raw_v,128);
    else h3_sol_accumulate<false>(reinterpret_cast<const float2 *>(q), 64, 0, tokens,
                      kc, vc, routes, first, stop, scale, workspace);
    for (int row = threadIdx.y; row < 64; row += blockDim.y) {
        float * correction = workspace + row*h3_sol_correction_stride;
        if (threadIdx.x == 0) {
            const float2 combined = h3_sol_merge_metadata(exact_meta[row], correction);
            correction[1] = combined.y;
        }
    }
    __syncthreads();
    for (int index = threadIdx.y*32 + threadIdx.x; index < 64*128; index += blockDim.y*32) {
        const int row = index/128, c = index%128;
        const float * correction = workspace + row*h3_sol_correction_stride;
        output[index] = (exact_values[index]*correction[2] + correction[4+c]*correction[3])/correction[1];
    }
}

static float rounded(float value) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

static void run_case(int tokens, int first, int stop, float scale, bool large_logits, bool keep_all, bool exact_terms = false) {
    std::printf("ORACLE exact_terms=%d\n",exact_terms);
    const int blocks = (tokens+63)/64;
    std::vector<float> q(64*128), kc(blocks*128), vc(blocks*128), exact_values(64*128), workspace(64*h3_sol_correction_stride);
    std::vector<int> routes(blocks);
    std::vector<float2> exact_meta(64);
    for (int row = 0; row < 64; ++row) {
        exact_meta[row] = make_float2(float(row%5)-2.f, .5f+.03f*row);
        for (int c = 0; c < 128; ++c) {
            q[row*128+c] = .15f*std::sin(float(7*row+3*c));
            exact_values[row*128+c] = .1f*std::cos(float(row+c));
        }
        if (large_logits) q[row*128] = (row%2 ? -13.f : 13.f)/scale;
    }
    for (int b = 0; b < blocks; ++b) {
        routes[b] = keep_all || b%3==0;
        const int count = std::min(64,tokens-64*b);
        for (int c = 0; c < 128; ++c) {
            kc[b*128+c] = rounded(.08f*std::cos(float(11*b+c)));
            vc[b*128+c] = count*rounded(.04f*std::sin(float(3*b+c)));
        }
        if (large_logits) kc[b*128] = b%2 ? -8.f : 8.f;
    }
    std::vector<ggml_fp16_t> raw_k(tokens*128),raw_v(tokens*128);
    for (int t=0;t<tokens;++t) for (int c=0;c<128;++c) {
        const int b=t/64,count=std::min(64,tokens-64*b);
        raw_k[t*128+c]=ggml_fp32_to_fp16(kc[b*128+c]+.02f*std::sin(float(t+c)));
        raw_v[t*128+c]=ggml_fp32_to_fp16(vc[b*128+c]/count+.01f*std::cos(float(t+3*c)));
    }
    auto * drk=upload(raw_k);auto * drv=upload(raw_v);
    auto * dq=upload(q);auto * dk=upload(kc);auto * dv=upload(vc);auto * dr=upload(routes);
    auto * dm=upload(exact_meta);auto * de=upload(exact_values);auto * dw=upload(workspace);
    float * output;check(cudaMalloc(&output,q.size()*sizeof(float)));
    correction_kernel<<<1,dim3(32,4)>>>(dq,dk,dv,dr,dm,de,dw,output,tokens,first,stop,scale,exact_terms,reinterpret_cast<const half *>(drk),reinterpret_cast<const half *>(drv));
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    std::vector<float> actual(q.size());check(cudaMemcpy(actual.data(),output,q.size()*sizeof(float),cudaMemcpyDeviceToHost));
    double error=0., energy=0., difference_without_correction=0.;float maximum_error=0.f;
    for (int row = 0; row < 64; ++row) {
        std::vector<std::vector<double>> scores(blocks);
        double maximum=exact_meta[row].x;
        for (int b=first+1;b<stop-1;++b) if (!routes[b]) {
            const int count=exact_terms ? std::min(64,tokens-64*b) : 1;scores[b].resize(count);
            for (int t=0;t<count;++t) {
                for (int c=0;c<128;++c) scores[b][t]+=double(rounded(rounded(q[row*128+c])*rounded(scale)))*(exact_terms ? ggml_fp16_to_fp32(raw_k[(b*64+t)*128+c]) : kc[b*128+c]);
                maximum=std::max(maximum,scores[b][t]+FATTN_KQ_MAX_OFFSET);
            }
        }
        const double exact_scale=std::exp(exact_meta[row].x-maximum);
        double mass=exact_meta[row].y*exact_scale;
        for (int b=first+1;b<stop-1;++b) if (!routes[b]) for (double score:scores[b]) mass+=std::exp(score-maximum)*(exact_terms ? 1 : std::min(64,tokens-64*b));
        for (int c=0;c<128;++c) {
            double numerator=exact_values[row*128+c]*exact_scale;
            for (int b=first+1;b<stop-1;++b) if (!routes[b]) for (int t=0;t<int(scores[b].size());++t)
                numerator+=std::exp(scores[b][t]-maximum)*(exact_terms ? ggml_fp16_to_fp32(raw_v[(b*64+t)*128+c]) : vc[b*128+c]);
            const double expected=numerator/mass,delta=actual[row*128+c]-expected;
            if (!std::isfinite(actual[row*128+c])) throw std::runtime_error("Nonfinite corrected output");
            maximum_error=std::max(maximum_error,float(std::abs(delta)));error+=delta*delta;energy+=expected*expected;
            difference_without_correction+=std::abs(expected-exact_values[row*128+c]/exact_meta[row].y);
        }
    }
    const double relative=std::sqrt(error/std::max(energy,1e-300));
    if (maximum_error>1e-4 || relative>1e-5) throw std::runtime_error("Centroid numerator/denominator correction differs from independent double oracle");
    if (!keep_all && difference_without_correction/q.size()<1e-3) throw std::runtime_error("Correction negative control was not exercised");
    void * allocations[] = {dq,dk,dv,dr,dm,de,dw,output,drk,drv};
    for (void * allocation:allocations) check(cudaFree(allocation));
    std::printf("PASS tokens=%d first=%d stop=%d scale=%.6g large_logits=%d all_keep=%d max_abs=%.9g relative_rms=%.9g\n",tokens,first,stop,scale,large_logits,keep_all,maximum_error,relative);
}

int main() {
    try {
        int count;check(cudaGetDeviceCount(&count));if (count!=1) return 77;
        for (bool large:{false,true}) {
            run_case(257,0,5,1.f,large,false,true);
            run_case(1025,3,11,128.f/std::sqrt(128.f),large,false,true);
            run_case(257,0,5,1.f,large,false);
            run_case(1025,3,11,128.f/std::sqrt(128.f),large,false);
            run_case(1025,0,17,.125f,large,true);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL %s\n",error.what());return 1;
    }
}
