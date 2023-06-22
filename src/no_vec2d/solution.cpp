#include "common.h"
#include "social_force.h"
#include <stdio.h>
#include <immintrin.h>

__m256d apply_exp(__m256d a) {
    double* elements = (double*)&a;
    for (int i = 0; i < 4; ++i) {
        elements[i] = exp(elements[i]);
    }
    return a;
}

// 4. AVX ++ if else removal
// somehow slower than 3 & 3.4.... need investigation
void optimized_4(int np, int num_runs){

    Vector2d vbs[n_p];
    double vb_norm_sqrs[n_p];
    #pragma ivdep
    for (int a = 0; a < n_p; a++){
        vbs[a] = v[a] * Dt;
        vb_norm_sqrs[a] = vbs[a].x*vbs[a].x+vbs[a].y*vbs[a].y;
    }

    double sigma_inv = 1 / sigma;
    double _const = vab0 * sigma_inv * 4;
    double _const1 = UaB0/R;
    __m256d _zero_point_five = _mm256_set1_pd(0.5);
    __m256d all_one = _mm256_set1_pd(1);
    __m256d all_zeros = _mm256_set1_pd(0.);
    __m256d all_const = _mm256_set1_pd( _const );
    __m256d all_inv_sigma = _mm256_set1_pd( - sigma_inv);
    __m256d all_cos_phi = _mm256_set1_pd(cos(phi));
    __m256d all_c = _mm256_set1_pd(c);
     for (int i=0; i<n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);
        
        // acceleration term
        Vector2d F0_alpha = (e[i]-v[i]) / tau_a * v0[i]; // 6 flops
        F_alpha += F0_alpha; // 2 flops

        Vector2d F_rep_ped(0., 0.);
        double norm_sum; double norm_sum1; double norm_sum2; double norm_sum3;

        // 1. load r[i] and setpd [r[i], r[i], r[i], r[i]], same for e[i]
        __m256d r_i_x = _mm256_broadcast_sd(&r_x[i]);
        __m256d r_i_y = _mm256_broadcast_sd(&r_y[i]);

        __m256d e_i_x = _mm256_broadcast_sd(&e_x[i]);
        __m256d e_i_y = _mm256_broadcast_sd(&e_y[i]);
        
        for (int j=0; j<n_p; j+=4) {

            // original
            // Vector2d rab = r[i] - r[j];
            // Vector2d rab1 = r[i] - r[j+1];
            // Vector2d rab2 = r[i] - r[j+2];
            // Vector2d rab3 = r[i] - r[j+3];

            // AVX
            __m256d r_j_x = _mm256_loadu_pd(&r_x[j]);
            __m256d r_j_y = _mm256_loadu_pd(&r_y[j]);
            __m256d rab_x = _mm256_sub_pd(r_i_x, r_j_x);
            __m256d rab_y = _mm256_sub_pd(r_i_y, r_j_y);

            // original          
            // Vector2d vb = v[j] * Dt;
            // Vector2d vb1 = v[j+1] * Dt;
            // Vector2d vb2 = v[j+2] * Dt;
            // Vector2d vb3 = v[j+3] * Dt;

            // AVX
            __m256d vb_x = _mm256_set_pd(vbs[j].x, vbs[j+1].x, vbs[j+2].x, vbs[j+3].x);
            __m256d vb_y = _mm256_set_pd(vbs[j].y, vbs[j+1].y, vbs[j+2].y, vbs[j+3].y);

            // original
            // Vector2d rabvb = rab - vb;
            // Vector2d rabvb1 = rab1 - vb1;
            // Vector2d rabvb2 = rab2 - vb2;
            // Vector2d rabvb3 = rab3 - vb3;

            // AVX
            __m256d rabvb_x = _mm256_sub_pd(rab_x, vb_x);
            __m256d rabvb_y = _mm256_sub_pd(rab_y, vb_y);

            //original
            // norm_sum = rab.norm()+rabvb.norm();
            // norm_sum1 = rab1.norm()+rabvb1.norm();
            // norm_sum2 = rab2.norm()+rabvb2.norm();
            // norm_sum3 = rab3.norm()+rabvb3.norm();

            // AVX
            __m256d vb_norm_sqr = _mm256_load_pd(&vb_norm_sqrs[j]);
            __m256d rab_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(rab_y, rab_y, _mm256_mul_pd(rab_x, rab_x)));
            __m256d rabvb_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(rabvb_y, rabvb_y, _mm256_mul_pd(rabvb_x, rabvb_x)));
            __m256d norm_sum = _mm256_add_pd(rab_norm, rabvb_norm);

            // Original
            // double b = sqrt(norm_sum*norm_sum - vb.norm()*vb.norm()) * 0.5 ; // 4x3+7=19
            // double b1 = sqrt(norm_sum1*norm_sum1 - vb1.norm()*vb1.norm()) * 0.5 ; // 4x3+7=19
            // double b2 = sqrt(norm_sum2*norm_sum2 - vb2.norm()*vb2.norm()) * 0.5 ; // 4x3+7=19
            // double b3 = sqrt(norm_sum3*norm_sum3 - vb3.norm()*vb3.norm()) * 0.5 ; // 4x3+7=19

            // AVX
            __m256d b = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_fmsub_pd(norm_sum, norm_sum, vb_norm_sqr)), _zero_point_five);

            // Original
            // double wef; double wef1; double wef2; double wef3;
            // Vector2d f_ab(0., 0.);
            // if  (rab.norm() != 0.){
            //     f_ab = _const*exp(-b/sigma) * norm_sum *(rab/rab.norm()+rabvb/rabvb.norm()) / b;  // 12
            // }

            // Vector2d f_ab1(0., 0.);
            // if  (rab1.norm() != 0.){
            //     f_ab1 = _const*exp(-b1/sigma) * norm_sum1 *(rab1/rab1.norm()+rabvb1/rabvb1.norm()) / b1;  // 12
            // }

            // Vector2d f_ab2(0., 0.);
            // if  (rab2.norm() != 0.){
            //     f_ab2 = _const*exp(-b2/sigma) * norm_sum2 *(rab2/rab2.norm()+rabvb2/rabvb2.norm()) / b2;  // 12
            // }

            // Vector2d f_ab3(0., 0.);
            // if  (rab3.norm() != 0.){
            //     f_ab3 = _const*exp(-b3/sigma) * norm_sum3 *(rab3/rab3.norm()+rabvb3/rabvb3.norm()) / b3;  // 12
            // }
            

            // AVX
            __m256d mask = _mm256_cmp_pd(rab_norm, all_zeros, _CMP_NEQ_OQ);
            __m256d expn = apply_exp(_mm256_mul_pd(b, all_inv_sigma));

            __m256d f_ab = _mm256_mul_pd( _mm256_mul_pd(all_const, expn ) , norm_sum);
            __m256d f_ab_x = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_x, rab_norm), _mm256_div_pd(rabvb_x, rabvb_norm)), f_ab);
            f_ab_x = _mm256_div_pd(f_ab_x, b);
            f_ab_x = _mm256_blendv_pd(all_zeros, f_ab_x, mask);
            __m256d f_ab_y = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_y, rab_norm), _mm256_div_pd(rabvb_y, rabvb_norm)), f_ab);
            f_ab_y = _mm256_div_pd(f_ab_y, b);
            f_ab_y = _mm256_blendv_pd(all_zeros, f_ab_y, mask);

            // Original
            // double wef; double wef1; double wef2; double wef3;
            // if (e[i].dot(-f_ab) >= -f_ab.norm()*cos(phi))
            //     wef = 1.;
            // else
            //     wef = c;

            // if (e[i].dot(-f_ab1) >= -f_ab1.norm()*cos(phi))
            //     wef1 = 1.;
            // else
            //     wef1 = c;

            // if (e[i].dot(-f_ab2) >= -f_ab2.norm()*cos(phi))
            //     wef2 = 1.;
            // else
            //     wef2 = c;
            
            // if (e[i].dot(-f_ab3) >= -f_ab3.norm()*cos(phi))
            //     wef3 = 1.;
            // else
            //     wef3 = c;

            // AVX
            __m256d dot_prod = _mm256_fmadd_pd(e_i_x, f_ab_x, _mm256_fmadd_pd(e_i_y, f_ab_y, all_zeros)); // e[i].dot(-f_ab)
            __m256d f_ab_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(f_ab_x, f_ab_x, _mm256_mul_pd(f_ab_y, f_ab_y)));
            __m256d comp = _mm256_sub_pd(all_zeros, _mm256_mul_pd(all_cos_phi, f_ab_norm));

             mask = _mm256_cmp_pd(dot_prod, comp, _CMP_GE_OQ);
            __m256d wef = _mm256_blendv_pd(all_c, all_one , mask);
            __m256d result_x = _mm256_mul_pd(wef, f_ab_x);
            __m256d result_y = _mm256_mul_pd(wef, f_ab_y);

            __m256d t1_x = _mm256_hadd_pd(result_x, result_x);
            __m128d t2_x = _mm256_extractf128_pd(t1_x, 1);
            __m128d t3_x = _mm_add_pd(_mm256_castpd256_pd128(t1_x), t2_x);

            __m256d t1_y = _mm256_hadd_pd(result_y, result_y);
            __m128d t2_y = _mm256_extractf128_pd(t1_y, 1);
            __m128d t3_y = _mm_add_pd(_mm256_castpd256_pd128(t1_y), t2_y);


            Vector2d ans(_mm_cvtsd_f64(t3_x), _mm_cvtsd_f64(t3_y));

            // F_rep_ped += wef * f_ab +  wef1 * f_ab1 +  wef2 * f_ab2 +  wef3 * f_ab3;
            F_rep_ped += ans;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = _const1*exp(-raB0.norm()/R) * raB0/raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y-height);
        Vector2d F_aB1 = _const1*exp(-raB1.norm()/R) * raB1/raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i]*dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha*dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }
}


void optimized_4_1(int np, int num_runs){

    Vector2d vbs[n_p];
    double vb_norm_sqrs[n_p];
    #pragma ivdep
    for (int a = 0; a < n_p; a++){
        vbs[a] = v[a] * Dt;
        vb_norm_sqrs[a] = vbs[a].x*vbs[a].x+vbs[a].y*vbs[a].y;
    }

    double sigma_inv = 1 / sigma;
    double _const = vab0 * sigma_inv * 4;
    double _const1 = UaB0/R;
    __m256d _zero_point_five = _mm256_set1_pd(0.5);
    __m256d all_one = _mm256_set1_pd(1);
    __m256d all_zeros = _mm256_set1_pd(0.);
    __m256d all_const = _mm256_set1_pd( _const );
    __m256d all_inv_sigma = _mm256_set1_pd( - sigma_inv);
    __m256d all_cos_phi = _mm256_set1_pd(cos(phi));
    __m256d all_c = _mm256_set1_pd(c);
     for (int i=0; i<n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);
        
        // acceleration term
        Vector2d F0_alpha = (e[i]-v[i]) / tau_a * v0[i]; // 6 flops
        F_alpha += F0_alpha; // 2 flops

        Vector2d F_rep_ped(0., 0.);
        double norm_sum; double norm_sum1; double norm_sum2; double norm_sum3;

        // 1. load r[i] and setpd [r[i], r[i], r[i], r[i]], same for e[i]
        __m256d r_i_x = _mm256_broadcast_sd(&r_x[i]);
        __m256d r_i_y = _mm256_broadcast_sd(&r_y[i]);

        __m256d e_i_x = _mm256_broadcast_sd(&e_x[i]);
        __m256d e_i_y = _mm256_broadcast_sd(&e_y[i]);
        
        for (int j=0; j<n_p; j+=4) {

            // original
            // Vector2d rab = r[i] - r[j];
            // Vector2d rab1 = r[i] - r[j+1];
            // Vector2d rab2 = r[i] - r[j+2];
            // Vector2d rab3 = r[i] - r[j+3];

            // AVX
            __m256d r_j_x = _mm256_loadu_pd(&r_x[j]);
            __m256d r_j_y = _mm256_loadu_pd(&r_y[j]);
            __m256d rab_x = _mm256_sub_pd(r_i_x, r_j_x);
            __m256d rab_y = _mm256_sub_pd(r_i_y, r_j_y);

            // original          
            // Vector2d vb = v[j] * Dt;
            // Vector2d vb1 = v[j+1] * Dt;
            // Vector2d vb2 = v[j+2] * Dt;
            // Vector2d vb3 = v[j+3] * Dt;

            // AVX
            __m256d vb_x = _mm256_set_pd(vbs[j].x, vbs[j+1].x, vbs[j+2].x, vbs[j+3].x);
            __m256d vb_y = _mm256_set_pd(vbs[j].y, vbs[j+1].y, vbs[j+2].y, vbs[j+3].y);

            // original
            // Vector2d rabvb = rab - vb;
            // Vector2d rabvb1 = rab1 - vb1;
            // Vector2d rabvb2 = rab2 - vb2;
            // Vector2d rabvb3 = rab3 - vb3;

            // AVX
            __m256d rabvb_x = _mm256_sub_pd(rab_x, vb_x);
            __m256d rabvb_y = _mm256_sub_pd(rab_y, vb_y);

            //original
            // norm_sum = rab.norm()+rabvb.norm();
            // norm_sum1 = rab1.norm()+rabvb1.norm();
            // norm_sum2 = rab2.norm()+rabvb2.norm();
            // norm_sum3 = rab3.norm()+rabvb3.norm();

            // AVX
            __m256d vb_norm_sqr = _mm256_load_pd(&vb_norm_sqrs[j]);
            __m256d rab_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(rab_y, rab_y, _mm256_mul_pd(rab_x, rab_x)));
            __m256d rabvb_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(rabvb_y, rabvb_y, _mm256_mul_pd(rabvb_x, rabvb_x)));
            __m256d norm_sum = _mm256_add_pd(rab_norm, rabvb_norm);

            // Original
            // double b = sqrt(norm_sum*norm_sum - vb.norm()*vb.norm()) * 0.5 ; // 4x3+7=19
            // double b1 = sqrt(norm_sum1*norm_sum1 - vb1.norm()*vb1.norm()) * 0.5 ; // 4x3+7=19
            // double b2 = sqrt(norm_sum2*norm_sum2 - vb2.norm()*vb2.norm()) * 0.5 ; // 4x3+7=19
            // double b3 = sqrt(norm_sum3*norm_sum3 - vb3.norm()*vb3.norm()) * 0.5 ; // 4x3+7=19

            // AVX
            __m256d b = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_fmsub_pd(norm_sum, norm_sum, vb_norm_sqr)), _zero_point_five);

            // Original
            // double wef; double wef1; double wef2; double wef3;
            // Vector2d f_ab(0., 0.);
            // if  (rab.norm() != 0.){
            //     f_ab = _const*exp(-b/sigma) * norm_sum *(rab/rab.norm()+rabvb/rabvb.norm()) / b;  // 12
            // }

            // Vector2d f_ab1(0., 0.);
            // if  (rab1.norm() != 0.){
            //     f_ab1 = _const*exp(-b1/sigma) * norm_sum1 *(rab1/rab1.norm()+rabvb1/rabvb1.norm()) / b1;  // 12
            // }

            // Vector2d f_ab2(0., 0.);
            // if  (rab2.norm() != 0.){
            //     f_ab2 = _const*exp(-b2/sigma) * norm_sum2 *(rab2/rab2.norm()+rabvb2/rabvb2.norm()) / b2;  // 12
            // }

            // Vector2d f_ab3(0., 0.);
            // if  (rab3.norm() != 0.){
            //     f_ab3 = _const*exp(-b3/sigma) * norm_sum3 *(rab3/rab3.norm()+rabvb3/rabvb3.norm()) / b3;  // 12
            // }
            

            // AVX
            __m256d mask = _mm256_cmp_pd(rab_norm, all_zeros, _CMP_NEQ_OQ);
            __m256d expn = _mm256_exp_pd(_mm256_mul_pd(b, all_inv_sigma));

            __m256d f_ab = _mm256_mul_pd( _mm256_mul_pd(all_const, expn ) , norm_sum);
            __m256d f_ab_x = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_x, rab_norm), _mm256_div_pd(rabvb_x, rabvb_norm)), f_ab);
            f_ab_x = _mm256_div_pd(f_ab_x, b);
            f_ab_x = _mm256_blendv_pd(all_zeros, f_ab_x, mask);
            __m256d f_ab_y = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_y, rab_norm), _mm256_div_pd(rabvb_y, rabvb_norm)), f_ab);
            f_ab_y = _mm256_div_pd(f_ab_y, b);
            f_ab_y = _mm256_blendv_pd(all_zeros, f_ab_y, mask);

            // Original
            // double wef; double wef1; double wef2; double wef3;
            // if (e[i].dot(-f_ab) >= f_ab.norm()*cos(phi))
            //     wef = 1.;
            // else
            //     wef = c;

            // if (e[i].dot(-f_ab1) >= f_ab1.norm()*cos(phi))
            //     wef1 = 1.;
            // else
            //     wef1 = c;

            // if (e[i].dot(-f_ab2) >= f_ab2.norm()*cos(phi))
            //     wef2 = 1.;
            // else
            //     wef2 = c;
            
            // if (e[i].dot(-f_ab3) >= f_ab3.norm()*cos(phi))
            //     wef3 = 1.;
            // else
            //     wef3 = c;

            // AVX
            __m256d dot_prod = _mm256_fmadd_pd(e_i_x, f_ab_x, _mm256_fmadd_pd(e_i_y, f_ab_y, all_zeros)); // e[i].dot(-f_ab)
            __m256d f_ab_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(f_ab_x, f_ab_x, _mm256_mul_pd(f_ab_y, f_ab_y)));
            __m256d comp = _mm256_mul_pd(all_cos_phi, f_ab_norm);

             mask = _mm256_cmp_pd(dot_prod, comp, _CMP_GE_OQ);
            __m256d wef = _mm256_blendv_pd(all_c, all_one , mask);
            __m256d result_x = _mm256_mul_pd(wef, f_ab_x);
            __m256d result_y = _mm256_mul_pd(wef, f_ab_y);

            __m256d t1_x = _mm256_hadd_pd(result_x, result_x);
            __m128d t2_x = _mm256_extractf128_pd(t1_x, 1);
            __m128d t3_x = _mm_add_pd(_mm256_castpd256_pd128(t1_x), t2_x);

            __m256d t1_y = _mm256_hadd_pd(result_y, result_y);
            __m128d t2_y = _mm256_extractf128_pd(t1_y, 1);
            __m128d t3_y = _mm_add_pd(_mm256_castpd256_pd128(t1_y), t2_y);


            Vector2d ans(_mm_cvtsd_f64(t3_x), _mm_cvtsd_f64(t3_y));

            // F_rep_ped += wef * f_ab +  wef1 * f_ab1 +  wef2 * f_ab2 +  wef3 * f_ab3;
            F_rep_ped += ans;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = _const1*exp(-raB0.norm()/R) * raB0/raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y-height);
        Vector2d F_aB1 = _const1*exp(-raB1.norm()/R) * raB1/raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i]*dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha*dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }
}



void optimized_4_2(int np, int num_runs){

    double* vbs_x = new double[n_p];
    double* vbs_y = new double[n_p];
    double vb_norm_sqrs[n_p];
    __m256d all_Dt = _mm256_set1_pd(Dt);\
    __m256d all_zeros = _mm256_set1_pd(0.);
    #pragma ivdep
    for (int a = 0; a < n_p; a+=4){
        //  vbs_x[a] = v_x[a] * Dt; vbs_y[a] = v_y[a] * Dt;

        _mm256_store_pd(&vbs_x[a], _mm256_mul_pd(_mm256_loadu_pd(&v_x[a]), all_Dt));
        _mm256_store_pd(&vbs_y[a], _mm256_mul_pd(_mm256_loadu_pd(&v_y[a]), all_Dt));

        // vb_norm_sqrs[a] =vbs_x[a]*vbs_x[a]+vbs_y[a]*vbs_y[a];
        __m256d _vbs_x = _mm256_loadu_pd(&vbs_x[a]);
        __m256d _vbs_y = _mm256_loadu_pd(&vbs_y[a]);
        _mm256_store_pd(&vb_norm_sqrs[a], _mm256_fmadd_pd(_vbs_x, _vbs_x, _mm256_fmadd_pd( _vbs_y,  _vbs_y, all_zeros)));
    }

    double sigma_inv = 1 / sigma;
    double _const = vab0 * sigma_inv * 4;
    double _const1 = UaB0/R;
    __m256d _zero_point_five = _mm256_set1_pd(0.5);
    __m256d all_one = _mm256_set1_pd(1);

    __m256d all_const = _mm256_set1_pd( _const );
    __m256d all_inv_sigma = _mm256_set1_pd( - sigma_inv);
    __m256d all_cos_phi = _mm256_set1_pd(cos(phi));
    __m256d all_c = _mm256_set1_pd(c);
     for (int i=0; i<n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);
        
        // acceleration term
        Vector2d F0_alpha = (e[i]-v[i]) / tau_a * v0[i]; // 6 flops
        F_alpha += F0_alpha; // 2 flops

        Vector2d F_rep_ped(0., 0.);
        double norm_sum; double norm_sum1; double norm_sum2; double norm_sum3;

        // 1. load r[i] and setpd [r[i], r[i], r[i], r[i]], same for e[i]
        __m256d r_i_x = _mm256_broadcast_sd(&r_x[i]);
        __m256d r_i_y = _mm256_broadcast_sd(&r_y[i]);

        __m256d e_i_x = _mm256_broadcast_sd(&e_x[i]);
        __m256d e_i_y = _mm256_broadcast_sd(&e_y[i]);
        
        for (int j=0; j<n_p; j+=4) {

            // original
            // Vector2d rab = r[i] - r[j];
            // Vector2d rab1 = r[i] - r[j+1];
            // Vector2d rab2 = r[i] - r[j+2];
            // Vector2d rab3 = r[i] - r[j+3];

            // AVX
            __m256d r_j_x = _mm256_loadu_pd(&r_x[j]);
            __m256d r_j_y = _mm256_loadu_pd(&r_y[j]);
            __m256d rab_x = _mm256_sub_pd(r_i_x, r_j_x);
            __m256d rab_y = _mm256_sub_pd(r_i_y, r_j_y);

            // original          
            // Vector2d vb = v[j] * Dt;
            // Vector2d vb1 = v[j+1] * Dt;
            // Vector2d vb2 = v[j+2] * Dt;
            // Vector2d vb3 = v[j+3] * Dt;

            // AVX
            __m256d vb_x = _mm256_loadu_pd(&vbs_x[j]);
            __m256d vb_y = _mm256_loadu_pd(&vbs_y[j]);

            // original
            // Vector2d rabvb = rab - vb;
            // Vector2d rabvb1 = rab1 - vb1;
            // Vector2d rabvb2 = rab2 - vb2;
            // Vector2d rabvb3 = rab3 - vb3;

            // AVX
            __m256d rabvb_x = _mm256_sub_pd(rab_x, vb_x);
            __m256d rabvb_y = _mm256_sub_pd(rab_y, vb_y);

            //original
            // norm_sum = rab.norm()+rabvb.norm();
            // norm_sum1 = rab1.norm()+rabvb1.norm();
            // norm_sum2 = rab2.norm()+rabvb2.norm();
            // norm_sum3 = rab3.norm()+rabvb3.norm();

            // AVX
            __m256d vb_norm_sqr = _mm256_load_pd(&vb_norm_sqrs[j]);
            __m256d rab_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(rab_y, rab_y, _mm256_mul_pd(rab_x, rab_x)));
            __m256d rabvb_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(rabvb_y, rabvb_y, _mm256_mul_pd(rabvb_x, rabvb_x)));
            __m256d norm_sum = _mm256_add_pd(rab_norm, rabvb_norm);

            // Original
            // double b = sqrt(norm_sum*norm_sum - vb.norm()*vb.norm()) * 0.5 ; // 4x3+7=19
            // double b1 = sqrt(norm_sum1*norm_sum1 - vb1.norm()*vb1.norm()) * 0.5 ; // 4x3+7=19
            // double b2 = sqrt(norm_sum2*norm_sum2 - vb2.norm()*vb2.norm()) * 0.5 ; // 4x3+7=19
            // double b3 = sqrt(norm_sum3*norm_sum3 - vb3.norm()*vb3.norm()) * 0.5 ; // 4x3+7=19

            // AVX
            __m256d b = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_fmsub_pd(norm_sum, norm_sum, vb_norm_sqr)), _zero_point_five);

            // Original
            // double wef; double wef1; double wef2; double wef3;
            // Vector2d f_ab(0., 0.);
            // if  (rab.norm() != 0.){
            //     f_ab = _const*exp(-b/sigma) * norm_sum *(rab/rab.norm()+rabvb/rabvb.norm()) / b;  // 12
            // }

            // Vector2d f_ab1(0., 0.);
            // if  (rab1.norm() != 0.){
            //     f_ab1 = _const*exp(-b1/sigma) * norm_sum1 *(rab1/rab1.norm()+rabvb1/rabvb1.norm()) / b1;  // 12
            // }

            // Vector2d f_ab2(0., 0.);
            // if  (rab2.norm() != 0.){
            //     f_ab2 = _const*exp(-b2/sigma) * norm_sum2 *(rab2/rab2.norm()+rabvb2/rabvb2.norm()) / b2;  // 12
            // }

            // Vector2d f_ab3(0., 0.);
            // if  (rab3.norm() != 0.){
            //     f_ab3 = _const*exp(-b3/sigma) * norm_sum3 *(rab3/rab3.norm()+rabvb3/rabvb3.norm()) / b3;  // 12
            // }
            

            // AVX
            __m256d mask = _mm256_cmp_pd(rab_norm, all_zeros, _CMP_NEQ_OQ);
            __m256d expn = _mm256_exp_pd(_mm256_mul_pd(b, all_inv_sigma));

            __m256d f_ab = _mm256_mul_pd( _mm256_mul_pd(all_const, expn ) , norm_sum);
            __m256d f_ab_x = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_x, rab_norm), _mm256_div_pd(rabvb_x, rabvb_norm)), f_ab);
            f_ab_x = _mm256_div_pd(f_ab_x, b);
            f_ab_x = _mm256_blendv_pd(all_zeros, f_ab_x, mask);
            __m256d f_ab_y = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_y, rab_norm), _mm256_div_pd(rabvb_y, rabvb_norm)), f_ab);
            f_ab_y = _mm256_div_pd(f_ab_y, b);
            f_ab_y = _mm256_blendv_pd(all_zeros, f_ab_y, mask);

            // Original
            // double wef; double wef1; double wef2; double wef3;
            // if (e[i].dot(-f_ab) >= f_ab.norm()*cos(phi))
            //     wef = 1.;
            // else
            //     wef = c;

            // if (e[i].dot(-f_ab1) >= f_ab1.norm()*cos(phi))
            //     wef1 = 1.;
            // else
            //     wef1 = c;

            // if (e[i].dot(-f_ab2) >= f_ab2.norm()*cos(phi))
            //     wef2 = 1.;
            // else
            //     wef2 = c;
            
            // if (e[i].dot(-f_ab3) >= f_ab3.norm()*cos(phi))
            //     wef3 = 1.;
            // else
            //     wef3 = c;

            // AVX
            __m256d dot_prod = _mm256_fmadd_pd(e_i_x, f_ab_x, _mm256_fmadd_pd(e_i_y, f_ab_y, all_zeros)); // e[i].dot(-f_ab)
            __m256d f_ab_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(f_ab_x, f_ab_x, _mm256_mul_pd(f_ab_y, f_ab_y)));
            __m256d comp = _mm256_mul_pd(all_cos_phi, f_ab_norm);

             mask = _mm256_cmp_pd(dot_prod, comp, _CMP_GE_OQ);
            __m256d wef = _mm256_blendv_pd(all_c, all_one , mask);
            __m256d result_x = _mm256_mul_pd(wef, f_ab_x);
            __m256d result_y = _mm256_mul_pd(wef, f_ab_y);

            __m256d t1_x = _mm256_hadd_pd(result_x, result_x);
            __m128d t2_x = _mm256_extractf128_pd(t1_x, 1);
            __m128d t3_x = _mm_add_pd(_mm256_castpd256_pd128(t1_x), t2_x);

            __m256d t1_y = _mm256_hadd_pd(result_y, result_y);
            __m128d t2_y = _mm256_extractf128_pd(t1_y, 1);
            __m128d t3_y = _mm_add_pd(_mm256_castpd256_pd128(t1_y), t2_y);


            Vector2d ans(_mm_cvtsd_f64(t3_x), _mm_cvtsd_f64(t3_y));

            // F_rep_ped += wef * f_ab +  wef1 * f_ab1 +  wef2 * f_ab2 +  wef3 * f_ab3;
            F_rep_ped += ans;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = _const1*exp(-raB0.norm()/R) * raB0/raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y-height);
        Vector2d F_aB1 = _const1*exp(-raB1.norm()/R) * raB1/raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i]*dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha*dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }
}


// 6 Outer Loop Unroll
// void optimized_6(int np, int num_runs) {
//     double _const = vab0 / sigma / 4;
//     double _const1 = UaB0 / R;
//     __m256d D_t = _mm256_set1_pd(Dt);
//     __m256d _zero_point_five = _mm256_set1_pd(0.5);
//     __m256d all_one = _mm256_set1_pd(1);
//     __m256d all_two = _mm256_set1_pd(2);
//     __m256d all_zeros = _mm256_set1_pd(0.);
//     __m256d all_const = _mm256_set1_pd(_const);
//     __m256d all_inv_sigma = _mm256_set1_pd(-1 / sigma);
//     __m256d all_cos_phi = _mm256_set1_pd(cos(phi));
//     __m256d all_c = _mm256_set1_pd(c);
//     __m256d all_inv_tau_a = _mm256_set1_pd(1 / tau_a);
//     __m256d all_height = _mm256_set1_pd(height);
//     __m256d all_inv_neg_R = _mm256_set1_pd(-1 / R);
//     __m256d all_const1 = _mm256_set1_pd(_const1);
//     __m256d all_dt = _mm256_set1_pd(dt);
//     for (int i = 0; i < n_p; i += 4) {

//         // // Original
//         // Vector2d F_alpha(.0, .0);
//         // // acceleration term
//         // Vector2d F0_alpha = (e[i]-v[i]) / tau_a * v0[i]; // 6 flops

//         // AVX
//         __m256d _F_alpha_x = _mm256_set1_pd(0);
//         __m256d _F_alpha_y = _mm256_set1_pd(0);

//         __m256d _e_x = _mm256_set_pd(e[i].x, e[i + 1].x, e[i + 2].x, e[i + 3].x);
//         __m256d _e_y = _mm256_set_pd(e[i].y, e[i + 1].y, e[i + 2].y, e[i + 3].y);

//         __m256d _v_x = _mm256_set_pd(v[i].x, v[i + 1].x, v[i + 2].x, v[i + 3].x);
//         __m256d _v_y = _mm256_set_pd(v[i].y, v[i + 1].y, v[i + 2].y, v[i + 3].y);

//         __m256d _v0 = _mm256_set_pd(v0[i], v0[i + 1], v0[i + 2], v0[i + 3]);
//         __m256d _F0_alpha_x = _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(_e_x, _v_x), all_inv_tau_a), _v0);
//         __m256d _F0_alpha_y = _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(_e_y, _v_y), all_inv_tau_a), _v0);

//         // // Original                
//         // F_alpha += F0_alpha; // 2 flops

//         // AVX
//         _F_alpha_x = _mm256_add_pd(_F_alpha_x, _F0_alpha_x);
//         _F_alpha_y = _mm256_add_pd(_F_alpha_y, _F0_alpha_y);

//         __m256d r_i_x = _mm256_broadcast_sd(&r[i].x);
//         __m256d r_i_y = _mm256_broadcast_sd(&r[i].y);

//         __m256d r_i_x1 = _mm256_broadcast_sd(&r[i + 1].x);
//         __m256d r_i_y1 = _mm256_broadcast_sd(&r[i + 1].y);

//         __m256d r_i_x2 = _mm256_broadcast_sd(&r[i + 2].x);
//         __m256d r_i_y2 = _mm256_broadcast_sd(&r[i + 2].y);

//         __m256d r_i_x3 = _mm256_broadcast_sd(&r[i + 3].x);
//         __m256d r_i_y3 = _mm256_broadcast_sd(&r[i + 3].y);

//         __m256d e_i_x = _mm256_broadcast_sd(&e[i].x);
//         __m256d e_i_y = _mm256_broadcast_sd(&e[i].y);

//         __m256d e_i_x1 = _mm256_broadcast_sd(&e[i + 1].x);
//         __m256d e_i_y1 = _mm256_broadcast_sd(&e[i + 1].y);

//         __m256d e_i_x2 = _mm256_broadcast_sd(&e[i + 2].x);
//         __m256d e_i_y2 = _mm256_broadcast_sd(&e[i + 2].y);

//         __m256d e_i_x3 = _mm256_broadcast_sd(&e[i + 3].x);
//         __m256d e_i_y3 = _mm256_broadcast_sd(&e[i + 3].y);

//         Vector2d F_rep_ped(0., 0.);
//         Vector2d F_rep_ped1(0., 0.);
//         Vector2d F_rep_ped2(0., 0.);
//         Vector2d F_rep_ped3(0., 0.);

//         double norm_sum; double norm_sum1; double norm_sum2; double norm_sum3;

//         // 0
//         for (int j = 0; j < n_p; j += 4) {

//             __m256d r_j_x = _mm256_set_pd(r[j].x, r[j + 1].x, r[j + 2].x, r[j + 3].x);
//             __m256d r_j_y = _mm256_set_pd(r[j].y, r[j + 1].y, r[j + 2].y, r[j + 3].y);
//             __m256d rab_x = _mm256_sub_pd(r_i_x, r_j_x);
//             __m256d rab_y = _mm256_sub_pd(r_i_y, r_j_y);


//             __m256d v_j_x = _mm256_set_pd(v[j].x, v[j + 1].x, v[j + 2].x, v[j + 3].x);
//             __m256d v_j_y = _mm256_set_pd(v[j].y, v[j + 1].y, v[j + 2].y, v[j + 3].y);
//             __m256d vb_x = _mm256_mul_pd(v_j_x, D_t);
//             __m256d vb_y = _mm256_mul_pd(v_j_y, D_t);

//             __m256d rabvb_x = _mm256_sub_pd(rab_x, vb_x);
//             __m256d rabvb_y = _mm256_sub_pd(rab_y, vb_y);


//             __m256d vb_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(vb_y, vb_y, _mm256_fmadd_pd(vb_x, vb_x, all_zeros)));
//             __m256d rab_norm_sum = _mm256_sqrt_pd(_mm256_fmadd_pd(rab_y, rab_y, _mm256_fmadd_pd(rab_x, rab_x, all_zeros)));
//             __m256d rabvb_norm_sum = _mm256_sqrt_pd(_mm256_fmadd_pd(rabvb_y, rabvb_y, _mm256_fmadd_pd(rabvb_x, rabvb_x, all_zeros)));
//             __m256d _norm_sum = _mm256_add_pd(rab_norm_sum, rabvb_norm_sum);


//             __m256d _b = _mm256_sqrt_pd(_mm256_fmsub_pd(_norm_sum, _norm_sum, _mm256_mul_pd(vb_norm, vb_norm)));
//             _b = _mm256_mul_pd(_b, _zero_point_five);

//             __m256d _mask = _mm256_cmp_pd(rab_norm_sum, all_zeros, _CMP_NEQ_OQ);

//             // AVX
//             __m256d expn = _mm256_exp_pd(_mm256_fmadd_pd(_b, all_inv_sigma, all_zeros));

//             __m256d _f_ab = _mm256_mul_pd(_mm256_mul_pd(all_const, expn), _norm_sum);
//             __m256d _f_ab_x = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_x, rab_norm_sum), _mm256_div_pd(rabvb_x, rabvb_norm_sum)), _f_ab);
//             _f_ab_x = _mm256_div_pd(_f_ab_x, _b);
//             _f_ab_x = _mm256_blendv_pd(all_zeros, _f_ab_x, _mask);
//             __m256d _f_ab_y = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_y, rab_norm_sum), _mm256_div_pd(rabvb_y, rabvb_norm_sum)), _f_ab);
//             _f_ab_y = _mm256_div_pd(_f_ab_y, _b);
//             _f_ab_y = _mm256_blendv_pd(all_zeros, _f_ab_y, _mask);

//             __m256d _f_ab_x_opp = _mm256_sub_pd(all_zeros, _f_ab_x);
//             __m256d _f_ab_y_opp = _mm256_sub_pd(all_zeros, _f_ab_y);


//             __m256d dot_prod = _mm256_fmadd_pd(e_i_x, _f_ab_x_opp, _mm256_fmadd_pd(e_i_y, _f_ab_y_opp, all_zeros)); // e[i].dot(-f_ab)
//             __m256d _f_ab_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(_f_ab_x, _f_ab_x, _mm256_mul_pd(_f_ab_y, _f_ab_y)));
//             __m256d comp = _mm256_sub_pd(all_zeros, _mm256_mul_pd(all_cos_phi, _f_ab_norm));

//             __m256d mask = _mm256_cmp_pd(dot_prod, comp, _CMP_GE_OQ);
//             __m256d _wef = _mm256_blendv_pd(all_c, all_one, mask);
//             __m256d result_x = _mm256_mul_pd(_wef, _f_ab_x);
//             __m256d result_y = _mm256_mul_pd(_wef, _f_ab_y);

//             __m256d t1_x = _mm256_hadd_pd(result_x, result_x);
//             __m128d t2_x = _mm256_extractf128_pd(t1_x, 1);
//             __m128d t3_x = _mm_add_pd(_mm256_castpd256_pd128(t1_x), t2_x);

//             __m256d t1_y = _mm256_hadd_pd(result_y, result_y);
//             __m128d t2_y = _mm256_extractf128_pd(t1_y, 1);
//             __m128d t3_y = _mm_add_pd(_mm256_castpd256_pd128(t1_y), t2_y);


//             Vector2d ans(_mm_cvtsd_f64(t3_x), _mm_cvtsd_f64(t3_y));
//             F_rep_ped += ans;
//         }

//         // 1 
//         for (int j = 0; j < n_p; j += 4) {

//             __m256d r_j_x = _mm256_set_pd(r[j].x, r[j + 1].x, r[j + 2].x, r[j + 3].x);
//             __m256d r_j_y = _mm256_set_pd(r[j].y, r[j + 1].y, r[j + 2].y, r[j + 3].y);
//             __m256d rab_x = _mm256_sub_pd(r_i_x1, r_j_x);
//             __m256d rab_y = _mm256_sub_pd(r_i_y1, r_j_y);


//             __m256d v_j_x = _mm256_set_pd(v[j].x, v[j + 1].x, v[j + 2].x, v[j + 3].x);
//             __m256d v_j_y = _mm256_set_pd(v[j].y, v[j + 1].y, v[j + 2].y, v[j + 3].y);
//             __m256d vb_x = _mm256_mul_pd(v_j_x, D_t);
//             __m256d vb_y = _mm256_mul_pd(v_j_y, D_t);

//             __m256d rabvb_x = _mm256_sub_pd(rab_x, vb_x);
//             __m256d rabvb_y = _mm256_sub_pd(rab_y, vb_y);


//             __m256d vb_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(vb_y, vb_y, _mm256_fmadd_pd(vb_x, vb_x, all_zeros)));
//             __m256d rab_norm_sum = _mm256_sqrt_pd(_mm256_fmadd_pd(rab_y, rab_y, _mm256_fmadd_pd(rab_x, rab_x, all_zeros)));
//             __m256d rabvb_norm_sum = _mm256_sqrt_pd(_mm256_fmadd_pd(rabvb_y, rabvb_y, _mm256_fmadd_pd(rabvb_x, rabvb_x, all_zeros)));
//             __m256d _norm_sum = _mm256_add_pd(rab_norm_sum, rabvb_norm_sum);


//             __m256d _b = _mm256_sqrt_pd(_mm256_fmsub_pd(_norm_sum, _norm_sum, _mm256_mul_pd(vb_norm, vb_norm)));
//             _b = _mm256_mul_pd(_b, _zero_point_five);

//             __m256d _mask = _mm256_cmp_pd(rab_norm_sum, all_zeros, _CMP_NEQ_OQ);

//             // AVX
//             __m256d expn = _mm256_exp_pd(_mm256_fmadd_pd(_b, all_inv_sigma, all_zeros));

//             __m256d _f_ab = _mm256_mul_pd(_mm256_mul_pd(all_const, expn), _norm_sum);
//             __m256d _f_ab_x = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_x, rab_norm_sum), _mm256_div_pd(rabvb_x, rabvb_norm_sum)), _f_ab);
//             _f_ab_x = _mm256_div_pd(_f_ab_x, _b);
//             _f_ab_x = _mm256_blendv_pd(all_zeros, _f_ab_x, _mask);
//             __m256d _f_ab_y = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_y, rab_norm_sum), _mm256_div_pd(rabvb_y, rabvb_norm_sum)), _f_ab);
//             _f_ab_y = _mm256_div_pd(_f_ab_y, _b);
//             _f_ab_y = _mm256_blendv_pd(all_zeros, _f_ab_y, _mask);

//             __m256d _f_ab_x_opp = _mm256_sub_pd(all_zeros, _f_ab_x);
//             __m256d _f_ab_y_opp = _mm256_sub_pd(all_zeros, _f_ab_y);


//             __m256d dot_prod = _mm256_fmadd_pd(e_i_x1, _f_ab_x_opp, _mm256_fmadd_pd(e_i_y1, _f_ab_y_opp, all_zeros)); // e[i].dot(-f_ab)
//             __m256d _f_ab_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(_f_ab_x, _f_ab_x, _mm256_mul_pd(_f_ab_y, _f_ab_y)));
//             __m256d comp = _mm256_sub_pd(all_zeros, _mm256_mul_pd(all_cos_phi, _f_ab_norm));

//             __m256d mask = _mm256_cmp_pd(dot_prod, comp, _CMP_GE_OQ);
//             __m256d _wef = _mm256_blendv_pd(all_c, all_one, mask);
//             __m256d result_x = _mm256_mul_pd(_wef, _f_ab_x);
//             __m256d result_y = _mm256_mul_pd(_wef, _f_ab_y);

//             __m256d t1_x = _mm256_hadd_pd(result_x, result_x);
//             __m128d t2_x = _mm256_extractf128_pd(t1_x, 1);
//             __m128d t3_x = _mm_add_pd(_mm256_castpd256_pd128(t1_x), t2_x);

//             __m256d t1_y = _mm256_hadd_pd(result_y, result_y);
//             __m128d t2_y = _mm256_extractf128_pd(t1_y, 1);
//             __m128d t3_y = _mm_add_pd(_mm256_castpd256_pd128(t1_y), t2_y);


//             Vector2d ans(_mm_cvtsd_f64(t3_x), _mm_cvtsd_f64(t3_y));
//             F_rep_ped1 += ans;
//         }

//         // 2 
//         for (int j = 0; j < n_p; j += 4) {

//             __m256d r_j_x = _mm256_set_pd(r[j].x, r[j + 1].x, r[j + 2].x, r[j + 3].x);
//             __m256d r_j_y = _mm256_set_pd(r[j].y, r[j + 1].y, r[j + 2].y, r[j + 3].y);
//             __m256d rab_x = _mm256_sub_pd(r_i_x2, r_j_x);
//             __m256d rab_y = _mm256_sub_pd(r_i_y2, r_j_y);


//             __m256d v_j_x = _mm256_set_pd(v[j].x, v[j + 1].x, v[j + 2].x, v[j + 3].x);
//             __m256d v_j_y = _mm256_set_pd(v[j].y, v[j + 1].y, v[j + 2].y, v[j + 3].y);
//             __m256d vb_x = _mm256_mul_pd(v_j_x, D_t);
//             __m256d vb_y = _mm256_mul_pd(v_j_y, D_t);

//             __m256d rabvb_x = _mm256_sub_pd(rab_x, vb_x);
//             __m256d rabvb_y = _mm256_sub_pd(rab_y, vb_y);


//             __m256d vb_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(vb_y, vb_y, _mm256_fmadd_pd(vb_x, vb_x, all_zeros)));
//             __m256d rab_norm_sum = _mm256_sqrt_pd(_mm256_fmadd_pd(rab_y, rab_y, _mm256_fmadd_pd(rab_x, rab_x, all_zeros)));
//             __m256d rabvb_norm_sum = _mm256_sqrt_pd(_mm256_fmadd_pd(rabvb_y, rabvb_y, _mm256_fmadd_pd(rabvb_x, rabvb_x, all_zeros)));
//             __m256d _norm_sum = _mm256_add_pd(rab_norm_sum, rabvb_norm_sum);


//             __m256d _b = _mm256_sqrt_pd(_mm256_fmsub_pd(_norm_sum, _norm_sum, _mm256_mul_pd(vb_norm, vb_norm)));
//             _b = _mm256_mul_pd(_b, _zero_point_five);

//             __m256d _mask = _mm256_cmp_pd(rab_norm_sum, all_zeros, _CMP_NEQ_OQ);

//             // AVX
//             __m256d expn = _mm256_exp_pd(_mm256_fmadd_pd(_b, all_inv_sigma, all_zeros));

//             __m256d _f_ab = _mm256_mul_pd(_mm256_mul_pd(all_const, expn), _norm_sum);
//             __m256d _f_ab_x = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_x, rab_norm_sum), _mm256_div_pd(rabvb_x, rabvb_norm_sum)), _f_ab);
//             _f_ab_x = _mm256_div_pd(_f_ab_x, _b);
//             _f_ab_x = _mm256_blendv_pd(all_zeros, _f_ab_x, _mask);
//             __m256d _f_ab_y = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_y, rab_norm_sum), _mm256_div_pd(rabvb_y, rabvb_norm_sum)), _f_ab);
//             _f_ab_y = _mm256_div_pd(_f_ab_y, _b);
//             _f_ab_y = _mm256_blendv_pd(all_zeros, _f_ab_y, _mask);

//             __m256d _f_ab_x_opp = _mm256_sub_pd(all_zeros, _f_ab_x);
//             __m256d _f_ab_y_opp = _mm256_sub_pd(all_zeros, _f_ab_y);


//             __m256d dot_prod = _mm256_fmadd_pd(e_i_x2, _f_ab_x_opp, _mm256_fmadd_pd(e_i_y2, _f_ab_y_opp, all_zeros)); // e[i].dot(-f_ab)
//             __m256d _f_ab_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(_f_ab_x, _f_ab_x, _mm256_mul_pd(_f_ab_y, _f_ab_y)));
//             __m256d comp = _mm256_sub_pd(all_zeros, _mm256_mul_pd(all_cos_phi, _f_ab_norm));

//             __m256d mask = _mm256_cmp_pd(dot_prod, comp, _CMP_GE_OQ);
//             __m256d _wef = _mm256_blendv_pd(all_c, all_one, mask);
//             __m256d result_x = _mm256_mul_pd(_wef, _f_ab_x);
//             __m256d result_y = _mm256_mul_pd(_wef, _f_ab_y);

//             __m256d t1_x = _mm256_hadd_pd(result_x, result_x);
//             __m128d t2_x = _mm256_extractf128_pd(t1_x, 1);
//             __m128d t3_x = _mm_add_pd(_mm256_castpd256_pd128(t1_x), t2_x);

//             __m256d t1_y = _mm256_hadd_pd(result_y, result_y);
//             __m128d t2_y = _mm256_extractf128_pd(t1_y, 1);
//             __m128d t3_y = _mm_add_pd(_mm256_castpd256_pd128(t1_y), t2_y);


//             Vector2d ans(_mm_cvtsd_f64(t3_x), _mm_cvtsd_f64(t3_y));
//             F_rep_ped2 += ans;
//         }

//         // 3 
//         for (int j = 0; j < n_p; j += 4) {

//             __m256d r_j_x = _mm256_set_pd(r[j].x, r[j + 1].x, r[j + 2].x, r[j + 3].x);
//             __m256d r_j_y = _mm256_set_pd(r[j].y, r[j + 1].y, r[j + 2].y, r[j + 3].y);
//             __m256d rab_x = _mm256_sub_pd(r_i_x2, r_j_x);
//             __m256d rab_y = _mm256_sub_pd(r_i_y2, r_j_y);


//             __m256d v_j_x = _mm256_set_pd(v[j].x, v[j + 1].x, v[j + 2].x, v[j + 3].x);
//             __m256d v_j_y = _mm256_set_pd(v[j].y, v[j + 1].y, v[j + 2].y, v[j + 3].y);
//             __m256d vb_x = _mm256_mul_pd(v_j_x, D_t);
//             __m256d vb_y = _mm256_mul_pd(v_j_y, D_t);

//             __m256d rabvb_x = _mm256_sub_pd(rab_x, vb_x);
//             __m256d rabvb_y = _mm256_sub_pd(rab_y, vb_y);


//             __m256d vb_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(vb_y, vb_y, _mm256_fmadd_pd(vb_x, vb_x, all_zeros)));
//             __m256d rab_norm_sum = _mm256_sqrt_pd(_mm256_fmadd_pd(rab_y, rab_y, _mm256_fmadd_pd(rab_x, rab_x, all_zeros)));
//             __m256d rabvb_norm_sum = _mm256_sqrt_pd(_mm256_fmadd_pd(rabvb_y, rabvb_y, _mm256_fmadd_pd(rabvb_x, rabvb_x, all_zeros)));
//             __m256d _norm_sum = _mm256_add_pd(rab_norm_sum, rabvb_norm_sum);


//             __m256d _b = _mm256_sqrt_pd(_mm256_fmsub_pd(_norm_sum, _norm_sum, _mm256_mul_pd(vb_norm, vb_norm)));
//             _b = _mm256_mul_pd(_b, _zero_point_five);

//             __m256d _mask = _mm256_cmp_pd(rab_norm_sum, all_zeros, _CMP_NEQ_OQ);

//             // AVX
//             __m256d expn = _mm256_exp_pd(_mm256_fmadd_pd(_b, all_inv_sigma, all_zeros));

//             __m256d _f_ab = _mm256_mul_pd(_mm256_mul_pd(all_const, expn), _norm_sum);
//             __m256d _f_ab_x = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_x, rab_norm_sum), _mm256_div_pd(rabvb_x, rabvb_norm_sum)), _f_ab);
//             _f_ab_x = _mm256_div_pd(_f_ab_x, _b);
//             _f_ab_x = _mm256_blendv_pd(all_zeros, _f_ab_x, _mask);
//             __m256d _f_ab_y = _mm256_mul_pd(_mm256_add_pd(_mm256_div_pd(rab_y, rab_norm_sum), _mm256_div_pd(rabvb_y, rabvb_norm_sum)), _f_ab);
//             _f_ab_y = _mm256_div_pd(_f_ab_y, _b);
//             _f_ab_y = _mm256_blendv_pd(all_zeros, _f_ab_y, _mask);

//             __m256d _f_ab_x_opp = _mm256_sub_pd(all_zeros, _f_ab_x);
//             __m256d _f_ab_y_opp = _mm256_sub_pd(all_zeros, _f_ab_y);


//             __m256d dot_prod = _mm256_fmadd_pd(e_i_x3, _f_ab_x_opp, _mm256_fmadd_pd(e_i_y3, _f_ab_y_opp, all_zeros)); // e[i].dot(-f_ab)
//             __m256d _f_ab_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(_f_ab_x, _f_ab_x, _mm256_mul_pd(_f_ab_y, _f_ab_y)));
//             __m256d comp = _mm256_sub_pd(all_zeros, _mm256_mul_pd(all_cos_phi, _f_ab_norm));

//             __m256d mask = _mm256_cmp_pd(dot_prod, comp, _CMP_GE_OQ);
//             __m256d _wef = _mm256_blendv_pd(all_c, all_one, mask);
//             __m256d result_x = _mm256_mul_pd(_wef, _f_ab_x);
//             __m256d result_y = _mm256_mul_pd(_wef, _f_ab_y);

//             __m256d t1_x = _mm256_hadd_pd(result_x, result_x);
//             __m128d t2_x = _mm256_extractf128_pd(t1_x, 1);
//             __m128d t3_x = _mm_add_pd(_mm256_castpd256_pd128(t1_x), t2_x);

//             __m256d t1_y = _mm256_hadd_pd(result_y, result_y);
//             __m128d t2_y = _mm256_extractf128_pd(t1_y, 1);
//             __m128d t3_y = _mm_add_pd(_mm256_castpd256_pd128(t1_y), t2_y);


//             Vector2d ans(_mm_cvtsd_f64(t3_x), _mm_cvtsd_f64(t3_y));
//             F_rep_ped3 += ans;
//         }



//         // // Original
//         // F_alpha += F_rep_ped; // 2 flops

//         // AVX
//         _F_alpha_x = _mm256_add_pd(_F_alpha_x, _mm256_set_pd(F_rep_ped.x, F_rep_ped1.x, F_rep_ped2.x, F_rep_ped3.x));
//         _F_alpha_y = _mm256_add_pd(_F_alpha_y, _mm256_set_pd(F_rep_ped.y, F_rep_ped1.y, F_rep_ped2.y, F_rep_ped3.y));

//         // // Original
//         // Vector2d raB0(0., r[i].y);
//         // Vector2d F_aB0 = _const1*exp(-raB0.norm()/R) * raB0/raB0.norm(); // 15 flops
//         // Vector2d raB1(0., r[i].y-height);
//         // Vector2d F_aB1 = _const1*exp(-raB1.norm()/R) * raB1/raB1.norm();

//         // AVX
//         __m256d _raB0_x = _mm256_set1_pd(0.);
//         __m256d _raB0_y = _mm256_set_pd(r[i].y, r[i + 1].y, r[i + 2].y, r[i + 3].y);
//         __m256d _raB1_x = _mm256_set1_pd(0.);
//         __m256d _raB1_y = _mm256_sub_pd(_raB0_y, all_height);

//         __m256d _F_aB0_x = _mm256_mul_pd(_mm256_mul_pd(all_const1, _mm256_exp_pd(_mm256_mul_pd(_raB0_x, all_inv_neg_R))), _raB0_x);
//         _F_aB0_x = _mm256_div_pd(_F_aB0_x, _mm256_sqrt_pd(_mm256_mul_pd(_raB0_x, _raB0_x)));

//         __m256d _F_aB0_y = _mm256_mul_pd(_mm256_mul_pd(all_const1, _mm256_exp_pd(_mm256_mul_pd(_raB0_y, all_inv_neg_R))), _raB0_y);
//         _F_aB0_y = _mm256_div_pd(_F_aB0_y, _mm256_sqrt_pd(_mm256_mul_pd(_raB0_y, _raB0_y)));


//         __m256d _F_aB1_x = _mm256_mul_pd(_mm256_mul_pd(all_const1, _mm256_exp_pd(_mm256_mul_pd(_raB1_x, all_inv_neg_R))), _raB1_x);
//         _F_aB1_x = _mm256_div_pd(_F_aB1_x, _mm256_sqrt_pd(_mm256_mul_pd(_raB1_x, _raB1_x)));

//         __m256d _F_aB1_y = _mm256_mul_pd(_mm256_mul_pd(all_const1, _mm256_exp_pd(_mm256_mul_pd(_raB1_y, all_inv_neg_R))), _raB1_y);
//         _F_aB1_y = _mm256_div_pd(_F_aB1_y, _mm256_sqrt_pd(_mm256_mul_pd(_raB1_y, _raB1_y)));

//         // // Original
//         // F_alpha += F_aB0 + F_aB1; // 4 flops

//         // AVX 
//         _F_alpha_x = _mm256_add_pd(_F_alpha_x, _mm256_add_pd(_F_aB0_x, _F_aB1_x));
//         _F_alpha_y = _mm256_add_pd(_F_alpha_y, _mm256_add_pd(_F_aB0_y, _F_aB1_y));

//         // // Original
//         // r_next[i] += v[i]*dt; // 4 flops
//         // Vector2d w_alpha = v[i] + F_alpha*dt; // 4 flops

//         // if  ( vMax[i] >= w_alpha.norm()){
//         //     v_next[i] = w_alpha;
//         // }
//         // else{
//         //     v_next[i] = w_alpha * vMax[i] / w_alpha.norm(); // 4+2+1=7 flops
//         // }


//         // // AVX
//         double* _r_next_x = &r_next_x[i];
//         double* _r_next_y = &r_next_y[i];
//         double* _vMax = &vMax[i];
//         __m256d __r_next_x = _mm256_loadu_pd(_r_next_x);
//         __m256d __r_next_y = _mm256_loadu_pd(_r_next_y);
//         __m256d __v_x = _mm256_loadu_pd(v_x);
//         __m256d __v_y = _mm256_loadu_pd(v_y);
//         __m256d __vMax = _mm256_loadu_pd(_vMax);

//         __r_next_x = _mm256_add_pd(_mm256_mul_pd(__v_x, all_dt), __r_next_x);
//         __r_next_y = _mm256_add_pd(_mm256_mul_pd(__v_y, all_dt), __r_next_y);
//         _mm256_store_pd(&r_next_x[i], __r_next_x);
//         _mm256_store_pd(&r_next_y[i], __r_next_y);

//         __m256d _w_alpha_x = _mm256_fmadd_pd(_F_alpha_x, all_dt, __v_x);
//         __m256d _w_alpha_y = _mm256_fmadd_pd(_F_alpha_y, all_dt, __v_y);
//         __m256d _w_alpha_norm = _mm256_sqrt_pd(_mm256_fmadd_pd(_w_alpha_x, _w_alpha_x, _mm256_fmadd_pd(_w_alpha_y, _w_alpha_y, all_zeros)));
//         __m256d _cmp = _mm256_cmp_pd(__vMax, _w_alpha_norm, _CMP_GE_OQ);

//         __m256d _v_next__x = _mm256_blendv_pd(_w_alpha_x, _mm256_div_pd(_mm256_mul_pd(__vMax, _w_alpha_x), _w_alpha_norm), _cmp);
//         __m256d _v_next__y = _mm256_blendv_pd(_w_alpha_y, _mm256_div_pd(_mm256_mul_pd(__vMax, _w_alpha_y), _w_alpha_norm), _cmp);
//         _mm256_store_pd(&v_next_x[i], _v_next__x);
//         _mm256_store_pd(&v_next_y[i], _v_next__y);


//     }
// }

void register_functions() {
    add_function(&optimized_4, "optimized_4");
    add_function(&optimized_4_1, "optimized_4_1");
    add_function(&optimized_4_2, "optimized_4_2");

}

void updateFile(string name) {
    std::ofstream file;
    file.open(name);
    update(file);
}

void init_comp(int n_p) {
    init(n_p);
}