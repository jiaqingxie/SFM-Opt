#include "social_force.h"
#include <cstdint>
#include <immintrin.h>

pair<Vector2d, Vector2d> baseline(pedestrian pedestrians[], Vector2d r_next[], Vector2d v_next[], int n_p){
    for (int i=0; i<n_p; i++) {
        // cout<<"------------------------"<<endl;
        // cout<<"Pedestrian "<<i<<": "<<endl;
        pedestrian *pi = &pedestrians[i];
        // cout<<"r"<<i<<": "<<pi->r<<endl;
        Vector2d F_alpha(.0, .0);
        
        // acceleration term
        Vector2d F0_alpha = Acceleration_term(pi); // 6 flops
        // cout<<"F0_alpha: "<<F0_alpha<<endl;

        F_alpha += F0_alpha; // 2 flops

        Vector2d F_ab = Repulsive_pedestrian(pi, pedestrians, n_p); // 52n flops
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_ab; // 2 flops

        Vector2d raB0(0., pi->r.y);
        Vector2d F_aB0 = Repulsive_border(raB0); // 15 flops
        Vector2d raB1(0., pi->r.y-height);
        Vector2d F_aB1 = Repulsive_border(raB1);
        // cout<<"F_aB: "<<endl<<F_aB0<<endl<<F_aB1<<endl;

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += pi->v*dt; // 4 flops
        Vector2d w_alpha = pi->v + F_alpha*dt; // 4 flops
        v_next[i] = w_alpha * g(pi->vMax, w_alpha.norm()); // 4+2+1=7 flops
        // if (v_next[i].norm()>2)
        //     cout<<v_next[i].norm()<<endl;
    }
    return make_pair(r_next[n_p], v_next[n_p]);
}


// inline procedure calls
// code motion
// strength reduction
pair<Vector2d, Vector2d> optimized_1(pedestrian pedestrians[], Vector2d r_next[], Vector2d v_next[], int n_p){
    double cos_phi = cos(100./180*M_PI);
    double tau_a_inv = 1 / tau_a;
    double sigma_inv = 1/ sigma;

    for (int i = 0; i < n_p; i++) {
            pedestrian *pi = &pedestrians[i];
            Vector2d F_alpha(.0, .0);
            
            // acceleration term
            Vector2d F0_alpha = (pi->v0 * pi->e - pi->v) * tau_a_inv;

            F_alpha += F0_alpha; 

            // repulsive pedestrian
            Vector2d F_ab(0., 0.);
            for (int j = 0; j < n_p; j++) {
                Vector2d f_ab(.0, .0);
                pedestrian *pb = &pedestrians[j];
                if (pi == pb) {
                    continue;
                }
                Vector2d rab = pi->r - pb->r;
                Vector2d vb = pb->v*Dt;
                Vector2d rabvb = rab - vb;
                double b = sqrt((rab.norm()+rabvb.norm())*(rab.norm()+rabvb.norm()) - vb.norm()*vb.norm()) * 0.5;

                f_ab = vab0*sigma_inv*exp(-b*sigma_inv) * (rab.norm()+rabvb.norm())*(rab/rab.norm()+rabvb/rabvb.norm()) / (4*b); 

                // weight
                double wef;
                if (pi->e.dot(-f_ab) >= -f_ab.norm()*cos_phi)
                    wef = 1.;
                else
                    wef = c;
                Vector2d F_ab_temp = wef * f_ab;
                F_ab += F_ab_temp;

            }
            F_alpha += F_ab; 

            Vector2d raB0(0., pi->r.y);
            Vector2d F_aB0 = UaB0/R*exp(-raB0.norm()/R) * raB0/raB0.norm(); 
            Vector2d raB1(0., pi->r.y-height);
            Vector2d F_aB1 = UaB0/R*exp(-raB1.norm()/R) * raB1/raB1.norm(); 

            F_alpha += F_aB0 + F_aB1; 

            r_next[i] += pi->v*dt;
            Vector2d w_alpha = pi->v + F_alpha*dt;
            v_next[i] = w_alpha * g(pi->vMax, w_alpha.norm());
        }
    return make_pair(r_next[n_p], v_next[n_p]);
}

// there are a lot of inverse sqrts in line 69- try fast inverse sqrt
// also func is compute bound if all data fits in LLC
pair<Vector2d, Vector2d> optimized_2(pedestrian pedestrians[], Vector2d r_next[], Vector2d v_next[], int n_p){
    double cos_phi = cos(100./180*M_PI);
    double tau_a_inv = 1 / tau_a;
    double sigma_inv = 1/ sigma;

    for (int i = 0; i < n_p; i++) {
            pedestrian *pi = &pedestrians[i];
            Vector2d F_alpha(.0, .0);
            
            // acceleration term
            Vector2d F0_alpha = (pi->v0 * pi->e - pi->v) * tau_a_inv;

            F_alpha += F0_alpha; 

            // repulsive pedestrian
            Vector2d F_ab(0., 0.);
            for (int j = 0; j < n_p; j++) {
                Vector2d f_ab(.0, .0);
                pedestrian *pb = &pedestrians[j];
                if (pi == pb) {
                    continue;
                }
                Vector2d rab = pi->r - pb->r;
                Vector2d vb = pb->v*Dt;
                Vector2d rabvb = rab - vb;


                // more common subexpression elimination
                double rab_norm_sqr = rab.x*rab.x+rab.y*rab.y;
                double rabvb_norm_sqr = rabvb.x*rabvb.x+rabvb.y*rabvb.y;
                double vb_norm_sqr = vb.x*vb.x+vb.y*vb.y;
                double rab_norm = sqrt(rab_norm_sqr);
                double rabvb_norm = sqrt(rabvb_norm_sqr);

                // Quake's inverse sqrt: https://stackoverflow.com/questions/11644441/fast-inverse-square-root-on-x64
                double rab_norm_inv = rab_norm_sqr;
                double x2 = rab_norm_inv * 0.5;
                std::int64_t x3 = *(std::int64_t *) &rab_norm_inv;
                // The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rab_norm_inv = *(double *) &x3;
                rab_norm_inv = rab_norm_inv * (1.5 - (x2 * rab_norm_inv * rab_norm_inv));

                double rabvb_norm_inv = rabvb_norm_sqr;
                x2 = rabvb_norm_inv * 0.5;
                x3 = *(std::int64_t *) &rabvb_norm_inv;
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rabvb_norm_inv = *(double *) &x3;
                rabvb_norm_inv = rabvb_norm_inv * (1.5 - (x2 * rabvb_norm_inv * rabvb_norm_inv));
                
                double b = sqrt((rab_norm+rabvb_norm)*(rab_norm+rabvb_norm) - vb_norm_sqr) * 0.5;
                f_ab = vab0*sigma_inv*exp(-b*sigma_inv) * (rab_norm+rabvb_norm)*(rab*rab_norm_inv+rabvb*rabvb_norm_inv) / (4*b);


                // weight
                double wef;
                if (pi->e.dot(-f_ab) >= -f_ab.norm()*cos_phi)
                    wef = 1.;
                else
                    wef = c;
                Vector2d F_ab_temp = wef * f_ab;
                F_ab += F_ab_temp;

            }
            F_alpha += F_ab; 

            Vector2d raB0(0., pi->r.y);
            Vector2d F_aB0 = UaB0/R*exp(-raB0.norm()/R) * raB0/raB0.norm(); 
            Vector2d raB1(0., pi->r.y-height);
            Vector2d F_aB1 = UaB0/R*exp(-raB1.norm()/R) * raB1/raB1.norm(); 

            F_alpha += F_aB0 + F_aB1; 

            r_next[i] += pi->v*dt;
            Vector2d w_alpha = pi->v + F_alpha*dt;
            v_next[i] = w_alpha * g(pi->vMax, w_alpha.norm());
        }
    return make_pair(r_next[n_p], v_next[n_p]);
}

// 1. Based on optimization 3, we have the further calcualtion improvements
// 1) line 150, repeat calculation of rab_norm + rabvb_norm
// 2) line 151, vab0 * sigma_inv / 4 =======> new variable
// 3) line 167, 169, pre-calculate UaB0/R, also consider pow(a, 1/R) where 1/R is pre-calculated

pair<Vector2d, Vector2d> optimized_3(pedestrian pedestrians[], Vector2d r_next[], Vector2d v_next[], int n_p){
    double cos_phi = cos(100./180*M_PI);
    double tau_a_inv = 1 / tau_a;
    double sigma_inv = 1/ sigma;
    double norm_sum = 0;
    double _const = vab0*sigma_inv * 0.25;
    double _const1 = UaB0/R;

    for (int i = 0; i < n_p; i++) {
            pedestrian *pi = &pedestrians[i];
            Vector2d F_alpha(.0, .0);
            
            // acceleration term
            Vector2d F0_alpha = (pi->v0 * pi->e - pi->v) * tau_a_inv;

            F_alpha += F0_alpha; 

            // repulsive pedestrian
            Vector2d F_ab(0., 0.);
            for (int j = 0; j < n_p; j++) {
                Vector2d f_ab(.0, .0);
                pedestrian *pb = &pedestrians[j];
                if (pi == pb) {
                    continue;
                }
                Vector2d rab = pi->r - pb->r;
                Vector2d vb = pb->v*Dt;
                Vector2d rabvb = rab - vb;


                // more common subexpression elimination
                double rab_norm_sqr = rab.x*rab.x+rab.y*rab.y;
                double rabvb_norm_sqr = rabvb.x*rabvb.x+rabvb.y*rabvb.y;
                double vb_norm_sqr = vb.x*vb.x+vb.y*vb.y;
                double rab_norm = sqrt(rab_norm_sqr);
                double rabvb_norm = sqrt(rabvb_norm_sqr);
                //Improvement 1
                norm_sum = rab_norm + rabvb_norm;

                // Quake's inverse sqrt: https://stackoverflow.com/questions/11644441/fast-inverse-square-root-on-x64
                double rab_norm_inv = rab_norm_sqr;
                double x2 = rab_norm_inv * 0.5;
                std::int64_t x3 = *(std::int64_t *) &rab_norm_inv;
                // The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rab_norm_inv = *(double *) &x3;
                rab_norm_inv = rab_norm_inv * (1.5 - (x2 * rab_norm_inv * rab_norm_inv));

                double rabvb_norm_inv = rabvb_norm_sqr;
                x2 = rabvb_norm_inv * 0.5;
                x3 = *(std::int64_t *) &rabvb_norm_inv;
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rabvb_norm_inv = *(double *) &x3;
                rabvb_norm_inv = rabvb_norm_inv * (1.5 - (x2 * rabvb_norm_inv * rabvb_norm_inv));
                
                double b = sqrt(norm_sum * norm_sum - vb_norm_sqr) * 0.5; //Improvement 1

                // Improvement 2 
                f_ab = _const*exp(-b*sigma_inv) * (rab_norm+rabvb_norm)*(rab*rab_norm_inv+rabvb*rabvb_norm_inv) / b; 

                // weight
                double wef;
                if (pi->e.dot(-f_ab) >= -f_ab.norm()*cos_phi)
                    wef = 1.;
                else
                    wef = c;
                Vector2d F_ab_temp = wef * f_ab;
                F_ab += F_ab_temp;

            }
            F_alpha += F_ab; 

            Vector2d raB0(0., pi->r.y);
            Vector2d F_aB0 = _const1 * exp(-raB0.norm()/R) * raB0/raB0.norm(); 
            Vector2d raB1(0., pi->r.y-height);
            Vector2d F_aB1 = _const1 * exp(-raB1.norm()/R) * raB1/raB1.norm(); 

            F_alpha += F_aB0 + F_aB1; 

            r_next[i] += pi->v*dt;
            Vector2d w_alpha = pi->v + F_alpha*dt;
            v_next[i] = w_alpha * g(pi->vMax, w_alpha.norm());
        }
    return make_pair(r_next[n_p], v_next[n_p]);
}

// unroll which might lead to AVX Intrinsics Implementation 
// Since we found the compiler didnt perform vectorization at all
// bug here: need to check where the compiler didnt perform vectorization
// First attempt is to directly unroll loops

// Additional: pi->r memory access
pair<Vector2d, Vector2d> optimized_4(pedestrian pedestrians[], Vector2d r_next[], Vector2d v_next[], int n_p){
    double cos_phi = cos(100./180*M_PI);
    double tau_a_inv = 1 / tau_a;
    double sigma_inv = 1/ sigma;
    double norm_sum = 0; double norm_sum_1 = 0; double norm_sum_2 = 0; double norm_sum_3 = 0;
    double _const = 2 * vab0*sigma_inv * 0.25;
    double _const1 = UaB0/R;
    int i = 0;
    for (; i < n_p; i+=1) {
            pedestrian *pi = &pedestrians[i];
            Vector2d F_alpha(.0, .0);
            // acceleration term
            Vector2d F0_alpha = (pi->v0 * pi->e - pi->v) * tau_a_inv;

            F_alpha += F0_alpha; 
            
            Vector2d pi_r = pi->r;
            double pi_r_y = pi_r.y;
            Vector2d pi_v = pi->v;

            // repulsive pedestrian
            Vector2d F_ab(0., 0.);
            for (int j = 0; j < n_p; j += 4) {
                Vector2d f_ab(.0, .0);
                Vector2d f_ab_1(.0, .0);
                Vector2d f_ab_2(.0, .0);
                Vector2d f_ab_3(.0, .0);
                pedestrian *pb = &pedestrians[j];
                pedestrian *pb_1 = &pedestrians[j+1];
                pedestrian *pb_2 = &pedestrians[j+2];
                pedestrian *pb_3 = &pedestrians[j+3];




                if (pi == pb) {
                     continue;
                }
                
                if (pi == pb_1)
                {
                    continue;
                }
                
                if (pi == pb_2)
                {
                    continue;
                }
                
                if (pi == pb_3)
                {
                    continue;
                }

                Vector2d rab = pi_r - pb->r;
                Vector2d rab_1 = pi_r - pb_1->r;
                Vector2d rab_2 = pi_r - pb_2->r;
                Vector2d rab_3 = pi_r - pb_3->r;

                Vector2d vb = pb->v*Dt;
                Vector2d vb_1 = pb_1->v*Dt;
                Vector2d vb_2 = pb_2->v*Dt;
                Vector2d vb_3 = pb_3->v*Dt;

                Vector2d rabvb = rab - vb;
                Vector2d rabvb_1 = rab_1 - vb_1;
                Vector2d rabvb_2 = rab_2 - vb_2;
                Vector2d rabvb_3 = rab_3 - vb_3;


                // more common subexpression elimination
                double rab_norm_sqr = rab.x*rab.x+rab.y*rab.y;
                double rabvb_norm_sqr = rabvb.x*rabvb.x+rabvb.y*rabvb.y;
                double vb_norm_sqr = vb.x*vb.x+vb.y*vb.y;
                double rab_norm = sqrt(rab_norm_sqr);
                double rabvb_norm = sqrt(rabvb_norm_sqr);

                double rab_norm_sqr_1 = rab_1.x*rab_1.x+rab_1.y*rab_1.y;
                double rabvb_norm_sqr_1 = rabvb_1.x*rabvb_1.x+rabvb_1.y*rabvb_1.y;
                double vb_norm_sqr_1 = vb_1.x*vb_1.x+vb_1.y*vb_1.y;
                double rab_norm_1 = sqrt(rab_norm_sqr_1);
                double rabvb_norm_1 = sqrt(rabvb_norm_sqr_1);

                double rab_norm_sqr_2 = rab_2.x*rab_2.x+rab_2.y*rab_2.y;
                double rabvb_norm_sqr_2 = rabvb_2.x*rabvb_2.x+rabvb_2.y*rabvb_2.y;
                double vb_norm_sqr_2 = vb_2.x*vb_2.x+vb_2.y*vb_2.y;
                double rab_norm_2 = sqrt(rab_norm_sqr_2);
                double rabvb_norm_2 = sqrt(rabvb_norm_sqr_2);

                double rab_norm_sqr_3 = rab_3.x*rab_3.x+rab_3.y*rab_3.y;
                double rabvb_norm_sqr_3 = rabvb_3.x*rabvb_3.x+rabvb_3.y*rabvb_3.y;
                double vb_norm_sqr_3 = vb_3.x*vb_3.x+vb_3.y*vb_3.y;
                double rab_norm_3 = sqrt(rab_norm_sqr_3);
                double rabvb_norm_3 = sqrt(rabvb_norm_sqr_3);

                //Improvement 
                norm_sum = rab_norm + rabvb_norm;
                norm_sum_1 = rab_norm_1 + rabvb_norm_1;
                norm_sum_2 = rab_norm_2 + rabvb_norm_2;
                norm_sum_3 = rab_norm_3 + rabvb_norm_3;

                double rab_norm_inv = rab_norm_sqr;
                double x2 = rab_norm_inv * 0.5;
                std::int64_t x3 = *(std::int64_t *) &rab_norm_inv;
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rab_norm_inv = *(double *) &x3;
                rab_norm_inv = rab_norm_inv * (1.5 - (x2 * rab_norm_inv * rab_norm_inv));

                double rab_norm_inv_1 = rab_norm_sqr_1;
                x2 = rab_norm_inv_1 * 0.5;
                x3 = *(std::int64_t *) &rab_norm_inv_1;
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rab_norm_inv_1 = *(double *) &x3;
                rab_norm_inv_1 = rab_norm_inv_1 * (1.5 - (x2 * rab_norm_inv_1 * rab_norm_inv_1));

                double rab_norm_inv_2 = rab_norm_sqr_2;
                x2 = rab_norm_inv_2 * 0.5;
                x3 = *(std::int64_t *) &rab_norm_inv_2;
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rab_norm_inv_2 = *(double *) &x3;
                rab_norm_inv_2 = rab_norm_inv_2 * (1.5 - (x2 * rab_norm_inv_2 * rab_norm_inv_2));

                double rab_norm_inv_3 = rab_norm_sqr_3;
                x2 = rab_norm_inv_3 * 0.5;
                x3 = *(std::int64_t *) &rab_norm_inv_3;
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rab_norm_inv_3 = *(double *) &x3;
                rab_norm_inv_3 = rab_norm_inv_3 * (1.5 - (x2 * rab_norm_inv_3 * rab_norm_inv_3));

                double rabvb_norm_inv = rabvb_norm_sqr;
                x2 = rabvb_norm_inv * 0.5;
                x3 = *(std::int64_t *) &rabvb_norm_inv;
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rabvb_norm_inv = *(double *) &x3;
                rabvb_norm_inv = rabvb_norm_inv * (1.5 - (x2 * rabvb_norm_inv * rabvb_norm_inv));

                double rabvb_norm_inv_1 = rabvb_norm_sqr_1;
                x2 = rabvb_norm_inv_1 * 0.5;
                x3 = *(std::int64_t *) &rabvb_norm_inv_1;
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rabvb_norm_inv_1 = *(double *) &x3;
                rabvb_norm_inv_1 = rabvb_norm_inv_1 * (1.5 - (x2 * rabvb_norm_inv_1* rabvb_norm_inv_1));

                double rabvb_norm_inv_2 = rabvb_norm_sqr_2;
                x2 = rabvb_norm_inv_2 * 0.5;
                x3 = *(std::int64_t *) &rabvb_norm_inv_2;
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rabvb_norm_inv_2 = *(double *) &x3;
                rabvb_norm_inv_2 = rabvb_norm_inv_2 * (1.5 - (x2 * rabvb_norm_inv_2* rabvb_norm_inv_2));


                double rabvb_norm_inv_3 = rabvb_norm_sqr_3;
                x2 = rabvb_norm_inv_3 * 0.5;
                x3 = *(std::int64_t *) &rabvb_norm_inv_3;
                x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
                rabvb_norm_inv_3 = *(double *) &x3;
                rabvb_norm_inv_3 = rabvb_norm_inv_3 * (1.5 - (x2 * rabvb_norm_inv_3* rabvb_norm_inv_3));

                double b = sqrt(norm_sum * norm_sum - vb_norm_sqr) ;
                double b_1 = sqrt(norm_sum_1 * norm_sum_1 - vb_norm_sqr_1) ;
                double b_2 = sqrt(norm_sum_2 * norm_sum_2 - vb_norm_sqr_2) ;
                double b_3 = sqrt(norm_sum_3 * norm_sum_3 - vb_norm_sqr_3) ;

                // Improvement 2 
                f_ab = _const*exp(-b*sigma_inv) * (rab_norm+rabvb_norm)*(rab*rab_norm_inv+rabvb*rabvb_norm_inv) / b; 
                f_ab_1 = _const*exp(-b_1*sigma_inv) * (rab_norm_1+rabvb_norm_1)*(rab_1*rab_norm_inv_1+rabvb_1*rabvb_norm_inv_1) / b_1;
                f_ab_2 = _const*exp(-b_2*sigma_inv) * (rab_norm_2+rabvb_norm_2)*(rab_2*rab_norm_inv_2+rabvb_2*rabvb_norm_inv_2) / b_2;
                f_ab_3 = _const*exp(-b_3*sigma_inv) * (rab_norm_3+rabvb_norm_3)*(rab_3*rab_norm_inv_3+rabvb_3*rabvb_norm_inv_3) / b_3;

                // weight
                double wef, wef1, wef2, wef3;
                if (pi->e.dot(-f_ab) >= -f_ab.norm()*cos_phi)
                    wef = 1.;
                else
                    wef = c;

                if (pi->e.dot(-f_ab_1) >= -f_ab_1.norm()*cos_phi)
                    wef1 = 1.;
                else
                    wef1 = c;

                if (pi->e.dot(-f_ab_2) >= -f_ab_2.norm()*cos_phi)
                    wef2 = 1.;
                else
                    wef2 = c;

                if (pi->e.dot(-f_ab_3) >= -f_ab_3.norm()*cos_phi)
                    wef3 = 1.;
                else
                    wef3 = c;
                F_ab += wef * f_ab  +   wef1 * f_ab_1 +  wef2 * f_ab_2 +  wef3 * f_ab_3;

            }
            F_alpha += F_ab; 

            Vector2d raB0(0., pi_r_y);
            Vector2d F_aB0 = _const1 * exp(-raB0.norm()/R) * raB0/raB0.norm(); 
            Vector2d raB1(0., pi_r_y-height);
            Vector2d F_aB1 = _const1 * exp(-raB1.norm()/R) * raB1/raB1.norm(); 

            F_alpha += F_aB0 + F_aB1; 

            r_next[i] += pi_v*dt;
            Vector2d w_alpha = pi_v + F_alpha*dt;
            v_next[i] = w_alpha * g(pi->vMax, w_alpha.norm());
        }
    return make_pair(r_next[n_p], v_next[n_p]);
}