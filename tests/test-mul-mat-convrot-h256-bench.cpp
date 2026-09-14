#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

template <typename T> static void rotate(T * x) {
    for (int s = 1; s < 256; s *= 4) {
        for (int b = 0; b < 256; b += 4*s) for (int i = 0; i < s; ++i) {
            T a=x[b+i], c=x[b+2*s+i], d=x[b+3*s+i], v=x[b+s+i];
            x[b+i]=(a+v+c-d)*T(0.5); x[b+s+i]=(a+v-c+d)*T(0.5);
            x[b+2*s+i]=(a-v+c+d)*T(0.5); x[b+3*s+i]=(-a+v+c+d)*T(0.5);
        }
    }
}

int main(int argc, char ** argv) {
    const int k=argc>1 ? std::atoi(argv[1]) : 5376;
    const int n=argc>2 ? std::atoi(argv[2]) : 14336;
    const int columns=argc>3 ? std::atoi(argv[3]) : 1;
    const bool half_input=argc>4 && std::atoi(argv[4]);
    if (k<=0 || k%256 || n<=0 || columns<=0) return 1;
    ggml_context * ctx=ggml_init({4*1024*1024,nullptr,true});
    ggml_backend_t backend=ggml_backend_cuda_init(0);
    if (!ctx || !backend) return 1;
    auto * x=ggml_new_tensor_2d(ctx,half_input?GGML_TYPE_F16:GGML_TYPE_F32,k,columns);
    auto * wi=ggml_new_tensor_2d(ctx,GGML_TYPE_I8,k,n);
    auto * s=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
    auto * wf=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
    auto * wh=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,k,n);
    auto * h=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,256);
    auto * gx=ggml_reshape_2d(ctx,half_input?ggml_cast(ctx,x,GGML_TYPE_F32):x,256,k/256*columns);
    auto * hx=ggml_mul_mat(ctx,h,gx);
    ggml_mul_mat_set_hint(hx,GGML_HINT_SRC0_IS_CONVROT_H256);
    auto * rx=ggml_reshape_2d(ctx,hx,k,columns);
    std::array<ggml_tensor *,4> outputs={ggml_mul_mat_convrot(ctx,x,wi,s,256),
        ggml_mul_mat(ctx,wf,rx),ggml_mul_mat(ctx,wh,rx),ggml_mul_mat(ctx,wh,rx)};
    ggml_mul_mat_set_prec(outputs[1],GGML_PREC_F32);
    ggml_mul_mat_set_prec(outputs[3],GGML_PREC_F32);
    const char * names[]={"native-i8","hint-f32","hint-f16-default","hint-f16-f32acc"};
    std::array<ggml_cgraph *,4> graphs;
    for(int j=0;j<4;++j){graphs[j]=ggml_new_graph(ctx);ggml_build_forward_expand(graphs[j],outputs[j]);}
    std::mt19937 rng(3923);
    std::uniform_real_distribution<float> a_dist(-1,1), s_dist(0.001f,0.03f);
    std::uniform_int_distribution<int> q_dist(-128,127);
    std::vector<float> xd(size_t(k)*columns), sd(n), wd(size_t(k)*n), hd(256*256);
    std::vector<ggml_fp16_t> xh(xd.size()), w16(wd.size());
    std::vector<int8_t> qd(wd.size());
    for(size_t i=0;i<xd.size();++i){xd[i]=a_dist(rng);xh[i]=ggml_fp32_to_fp16(xd[i]);if(half_input)xd[i]=ggml_fp16_to_fp32(xh[i]);}
    for(int row=0;row<n;++row){sd[row]=s_dist(rng);for(int i=0;i<k;++i){size_t at=size_t(row)*k+i;qd[at]=q_dist(rng);wd[at]=qd[at]*sd[row];w16[at]=ggml_fp32_to_fp16(wd[at]);}}
    for(int row=0;row<256;++row){std::array<float,256> b={};b[row]=1;rotate(b.data());std::copy(b.begin(),b.end(),hd.begin()+row*256);}
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)return 2;
    ggml_backend_tensor_set(x,half_input?(const void*)xh.data():(const void*)xd.data(),0,ggml_nbytes(x));
    ggml_backend_tensor_set(wi,qd.data(),0,ggml_nbytes(wi));ggml_backend_tensor_set(s,sd.data(),0,ggml_nbytes(s));
    ggml_backend_tensor_set(wf,wd.data(),0,ggml_nbytes(wf));ggml_backend_tensor_set(wh,w16.data(),0,ggml_nbytes(wh));
    ggml_backend_tensor_set(h,hd.data(),0,ggml_nbytes(h));
    // Independent double reference: rotate WEIGHTS, never the activations.
    // Validate 64 spread-out rows of every column (all rows for small tests).
    std::vector<int> rows;for(int j=0;j<std::min(n,64);++j)rows.push_back(j*(n-1)/std::max(1,std::min(n,64)-1));
    std::vector<double> refs(rows.size()*columns), bounds(refs.size());
    for(size_t r=0;r<rows.size();++r){std::vector<double>w(k);for(int i=0;i<k;++i)w[i]=double(qd[size_t(rows[r])*k+i])*sd[rows[r]];
        for(int i=0;i<k;i+=256)rotate(w.data()+i);
        for(int c=0;c<columns;++c){double sum=0,bound=0;for(int i=0;i<k;++i){double t=w[i]*xd[size_t(c)*k+i];sum+=t;bound+=std::abs(t);}refs[r+c*rows.size()]=sum;bounds[r+c*rows.size()]=bound;}}
    bool passed=true;
    std::printf("shape K=%d N=%d columns=%d input=%s; timings include backend synchronization\n",k,n,columns,half_input?"f16":"f32");
    for(int j=0;j<4;++j){if(ggml_backend_graph_compute(backend,graphs[j])!=GGML_STATUS_SUCCESS)return 3;
        std::vector<float> out(size_t(n)*columns);ggml_backend_tensor_get(outputs[j],out.data(),0,ggml_nbytes(outputs[j]));
        double max_abs=0,sq=0,denom=0;int fails=0;
        for(int c=0;c<columns;++c)for(size_t r=0;r<rows.size();++r){size_t t=r+c*rows.size();double got=out[rows[r]+size_t(c)*n],err=std::abs(got-refs[t]);
            max_abs=std::max(max_abs,err);sq+=err*err;denom+=refs[t]*refs[t];
            // Fixed error bound, chosen before measuring: accumulated FP32
            // roundoff scaled by sum(abs(products)); NaNs always fail.
            if(!std::isfinite(got)||err>2e-6*bounds[t]+1e-5)++fails;
        }
        if(j==0&&fails)passed=false;
        std::printf("%s accuracy max_abs=%.8g relative_l2=%.8g failures=%d/%zu\n",names[j],max_abs,std::sqrt(sq/std::max(denom,1e-30)),fails,refs.size());
    }
    std::array<std::vector<double>,4> timings;
    for(int j=0;j<4;++j)for(int i=0;i<5;++i)if(ggml_backend_graph_compute(backend,graphs[j])!=GGML_STATUS_SUCCESS)return 3;
    for(int trial=0;trial<7;++trial)for(int p=0;p<4;++p){int j=(p+trial)%4;ggml_backend_synchronize(backend);
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<30;++i)if(ggml_backend_graph_compute(backend,graphs[j])!=GGML_STATUS_SUCCESS)return 3;
        ggml_backend_synchronize(backend);timings[j].push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/30);
    }
    for(int j=0;j<4;++j){auto &t=timings[j];std::sort(t.begin(),t.end());std::printf("%s median_ms=%.6f min_ms=%.6f max_ms=%.6f\n",names[j],t[3],t.front(),t.back());}
    std::printf("native_weights_MiB=%.6f fp16_weights_MiB=%.6f fp32_weights_MiB=%.6f native_scratch_KiB=%.3f\n",(double(k)*n+double(n)*4)/1048576,double(k)*n*2/1048576,double(k)*n*4/1048576,double(k)*columns*4/1024);
    ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_free(ctx);return passed?0:4;
}
