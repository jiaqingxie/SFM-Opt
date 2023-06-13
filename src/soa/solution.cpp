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

void baseline(int np, int num_runs) {
    for (int i = 0; i < n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);

        // acceleration term
        Vector2d F0_alpha = Acceleration_term(i); // 6 flops
        // cout<<"F0_alpha: "<<F0_alpha<<endl;

        F_alpha += F0_alpha; // 2 flops

        Vector2d F_ab = Repulsive_pedestrian(i); // 52n flops
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_ab; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = Repulsive_border(raB0); // 15 flops
        Vector2d raB1(0., r[i].y - height);
        Vector2d F_aB1 = Repulsive_border(raB1);

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i] * dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha * dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }

}

// 1. inline procedure calls
void optimized_1(int np, int num_runs) {
    for (int i = 0; i < n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);

        // acceleration term
        Vector2d F0_alpha = (e[i] - v[i]) / tau_a * v0[i]; // 6 flops
        // cout<<"F0_alpha: "<<F0_alpha<<endl;

        F_alpha += F0_alpha; // 2 flops


        Vector2d F_rep_ped(0., 0.);

        for (int j = 0; j < n_p; j++) {
            if (i == j) {
                continue;
            }
            Vector2d rab = r[i] - r[j];
            Vector2d vb = v[j] * Dt;
            Vector2d rabvb = rab - vb;

            double b = sqrt((rab.norm() + rabvb.norm()) * (rab.norm() + rabvb.norm()) - vb.norm() * vb.norm()) / 2; // 4x3+7=19


            Vector2d f_ab = vab0 / sigma * exp(-b / sigma) * (rab.norm() + rabvb.norm()) * (rab / rab.norm() + rabvb / rabvb.norm()) / (4 * b); // 12

            double wef = w(e[i], -f_ab); // 11
            Vector2d F_ab = wef * f_ab;
            F_rep_ped += F_ab;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = UaB0 / R * exp(-raB0.norm() / R) * raB0 / raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y - height);
        Vector2d F_aB1 = UaB0 / R * exp(-raB1.norm() / R) * raB1 / raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i] * dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha * dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }
}

// 2. remove repeat computations, add pre-calculated constants
void optimized_2(int np, int num_runs) {
    double _const = vab0 / sigma / 4;
    double _const1 = UaB0 / R;
    for (int i = 0; i < n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);

        // acceleration term
        Vector2d F0_alpha = (e[i] - v[i]) / tau_a * v0[i]; // 6 flops


        F_alpha += F0_alpha; // 2 flops


        Vector2d F_rep_ped(0., 0.);
        double norm_sum;

        for (int j = 0; j < n_p; j++) {
            if (i == j) {
                continue;
            }
            Vector2d rab = r[i] - r[j];
            Vector2d vb = v[j] * Dt;
            Vector2d rabvb = rab - vb;

            norm_sum = rab.norm() + rabvb.norm();
            double b = sqrt(norm_sum * norm_sum - vb.norm() * vb.norm()) * 0.5; // 4x3+7=19

            Vector2d f_ab = _const * exp(-b / sigma) * norm_sum * (rab / rab.norm() + rabvb / rabvb.norm()) / b; // 12

            double wef;
            if (e[i].dot(-f_ab) >= f_ab.norm() * cos(phi))
                wef = 1.;
            else
                wef = c;
            Vector2d F_ab = wef * f_ab;
            F_rep_ped += F_ab;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = _const1 * exp(-raB0.norm() / R) * raB0 / raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y - height);
        Vector2d F_aB1 = _const1 * exp(-raB1.norm() / R) * raB1 / raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i] * dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha * dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }

}



// 2.5. strength reduction, more common subexp elim, fast inv sqrt
void optimized_2_5(int np, int num_runs) {
    double cos_phi = cos(100. / 180 * M_PI);
    double tau_a_inv = 1 / tau_a;
    double sigma_inv = 1 / sigma;
    double norm_sum = 0;
    double _const = vab0 * sigma_inv * 0.25;
    double _const1 = UaB0 / R;
    for (int i = 0; i < n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);

        // acceleration term
        Vector2d F0_alpha = (e[i] - v[i]) * tau_a_inv * v0[i]; // 6 flops


        F_alpha += F0_alpha; // 2 flops


        Vector2d F_rep_ped(0., 0.);
        double norm_sum;

        for (int j = 0; j < n_p; j++) {
            if (i == j) {
                continue;
            }
            Vector2d rab = r[i] - r[j];
            Vector2d vb = v[j] * Dt;
            Vector2d rabvb = rab - vb;

            // common subexp elim
            double rab_norm_sqr = rab.x * rab.x + rab.y * rab.y;
            double rabvb_norm_sqr = rabvb.x * rabvb.x + rabvb.y * rabvb.y;
            double vb_norm_sqr = vb.x * vb.x + vb.y * vb.y;
            double rab_norm = sqrt(rab_norm_sqr);
            double rabvb_norm = sqrt(rabvb_norm_sqr);

            norm_sum = rab_norm + rabvb_norm;
            double b = sqrt(norm_sum * norm_sum - vb_norm_sqr) * 0.5; // 4x3+7=19

            double rab_norm_inv = rab_norm_sqr;
            double x2 = rab_norm_inv * 0.5;
            std::int64_t x3 = *(std::int64_t*)&rab_norm_inv;
            x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
            rab_norm_inv = *(double*)&x3;
            rab_norm_inv = rab_norm_inv * (1.5 - (x2 * rab_norm_inv * rab_norm_inv));

            double rabvb_norm_inv = rabvb_norm_sqr;
            x2 = rabvb_norm_inv * 0.5;
            x3 = *(std::int64_t*)&rabvb_norm_inv;
            x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
            rabvb_norm_inv = *(double*)&x3;
            rabvb_norm_inv = rabvb_norm_inv * (1.5 - (x2 * rabvb_norm_inv * rabvb_norm_inv));

            Vector2d f_ab = _const * exp(-b * sigma_inv) * norm_sum * (rab * rab_norm_inv + rabvb * rabvb_norm_inv) / b; // 12

            double wef;
            if (e[i].dot(-f_ab) >= -f_ab.norm() * cos_phi)
                wef = 1.;
            else
                wef = c;
            Vector2d F_ab = wef * f_ab;
            F_rep_ped += F_ab;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = _const1 * exp(-raB0.norm() / R) * raB0 / raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y - height);
        Vector2d F_aB1 = _const1 * exp(-raB1.norm() / R) * raB1 / raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i] * dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha * dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }
}

// 3. unroll loops
void optimized_3(int np, int num_runs) {
    double _const = vab0 / sigma / 4;
    double _const1 = UaB0 / R;
    double flag0 = 1; double flag1 = 1; double flag2 = 1; double flag3 = 1;
    for (int i = 0; i < n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);

        // acceleration term
        Vector2d F0_alpha = (e[i] - v[i]) / tau_a * v0[i]; // 6 flops


        F_alpha += F0_alpha; // 2 flops


        Vector2d F_rep_ped(0., 0.);
        double norm_sum; double norm_sum1; double norm_sum2; double norm_sum3;

        for (int j = 0; j < n_p; j += 4) {

            Vector2d rab = r[i] - r[j];
            Vector2d rab1 = r[i] - r[j + 1];
            Vector2d rab2 = r[i] - r[j + 2];
            Vector2d rab3 = r[i] - r[j + 3];

            Vector2d vb = v[j] * Dt;
            Vector2d vb1 = v[j + 1] * Dt;
            Vector2d vb2 = v[j + 2] * Dt;
            Vector2d vb3 = v[j + 3] * Dt;

            Vector2d rabvb = rab - vb;
            Vector2d rabvb1 = rab1 - vb1;
            Vector2d rabvb2 = rab2 - vb2;
            Vector2d rabvb3 = rab3 - vb3;

            norm_sum = rab.norm() + rabvb.norm();
            norm_sum1 = rab1.norm() + rabvb1.norm();
            norm_sum2 = rab2.norm() + rabvb2.norm();
            norm_sum3 = rab3.norm() + rabvb3.norm();


            double b = sqrt(norm_sum * norm_sum - vb.norm() * vb.norm()) * 0.5; // 4x3+7=19
            double b1 = sqrt(norm_sum1 * norm_sum1 - vb1.norm() * vb1.norm()) * 0.5; // 4x3+7=19
            double b2 = sqrt(norm_sum2 * norm_sum2 - vb2.norm() * vb2.norm()) * 0.5; // 4x3+7=19
            double b3 = sqrt(norm_sum3 * norm_sum3 - vb3.norm() * vb3.norm()) * 0.5; // 4x3+7=19

            double wef; double wef1; double wef2; double wef3;
            Vector2d f_ab(0., 0.);
            if (rab.norm() != 0.) {
                f_ab = _const * exp(-b / sigma) * norm_sum * (rab / rab.norm() + rabvb / rabvb.norm()) / b;  // 12
            }

            Vector2d f_ab1(0., 0.);
            if (rab1.norm() != 0.) {
                f_ab1 = _const * exp(-b1 / sigma) * norm_sum1 * (rab1 / rab1.norm() + rabvb1 / rabvb1.norm()) / b1;  // 12
            }

            Vector2d f_ab2(0., 0.);
            if (rab2.norm() != 0.) {
                f_ab2 = _const * exp(-b2 / sigma) * norm_sum2 * (rab2 / rab2.norm() + rabvb2 / rabvb2.norm()) / b2;  // 12
            }

            Vector2d f_ab3(0., 0.);
            if (rab3.norm() != 0.) {
                f_ab3 = _const * exp(-b3 / sigma) * norm_sum3 * (rab3 / rab3.norm() + rabvb3 / rabvb3.norm()) / b3;  // 12
            }

            if (e[i].dot(-f_ab) >= f_ab.norm() * cos(phi))
                wef = 1.;
            else
                wef = c;

            if (e[i].dot(-f_ab1) >= f_ab1.norm() * cos(phi))
                wef1 = 1.;
            else
                wef1 = c;

            if (e[i].dot(-f_ab2) >= f_ab2.norm() * cos(phi))
                wef2 = 1.;
            else
                wef2 = c;

            if (e[i].dot(-f_ab3) >= f_ab3.norm() * cos(phi))
                wef3 = 1.;
            else
                wef3 = c;

            F_rep_ped += wef * f_ab + wef1 * f_ab1 + wef2 * f_ab2 + wef3 * f_ab3;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = _const1 * exp(-raB0.norm() / R) * raB0 / raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y - height);
        Vector2d F_aB1 = _const1 * exp(-raB1.norm() / R) * raB1 / raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i] * dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha * dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }
}




// 3.4 like 3.5 but no fast inv sqrt
// this is the fastest, non-avx impl so far
void optimized_3_4(int np, int num_runs) {
    double cos_phi = cos(100. / 180 * M_PI);
    double tau_a_inv = 1 / tau_a;
    double sigma_inv = 1 / sigma;
    double norm_sum = 0;
    double _const = vab0 * sigma_inv * 0.25;
    double _const1 = UaB0 / R;
    double flag0 = 1; double flag1 = 1; double flag2 = 1; double flag3 = 1;
    for (int i = 0; i < n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);

        // acceleration term
        Vector2d F0_alpha = (e[i] - v[i]) / tau_a * v0[i]; // 6 flops


        F_alpha += F0_alpha; // 2 flops


        Vector2d F_rep_ped(0., 0.);
        double norm_sum; double norm_sum1; double norm_sum2; double norm_sum3;

        for (int j = 0; j < n_p; j += 4) {

            Vector2d rab = r[i] - r[j];
            Vector2d rab1 = r[i] - r[j + 1];
            Vector2d rab2 = r[i] - r[j + 2];
            Vector2d rab3 = r[i] - r[j + 3];

            Vector2d vb = v[j] * Dt;
            Vector2d vb1 = v[j + 1] * Dt;
            Vector2d vb2 = v[j + 2] * Dt;
            Vector2d vb3 = v[j + 3] * Dt;

            Vector2d rabvb = rab - vb;
            Vector2d rabvb1 = rab1 - vb1;
            Vector2d rabvb2 = rab2 - vb2;
            Vector2d rabvb3 = rab3 - vb3;


            double rab_norm_sqr = rab.x * rab.x + rab.y * rab.y;
            double rabvb_norm_sqr = rabvb.x * rabvb.x + rabvb.y * rabvb.y;
            double vb_norm_sqr = vb.x * vb.x + vb.y * vb.y;
            double rab_norm = sqrt(rab_norm_sqr);
            double rabvb_norm = sqrt(rabvb_norm_sqr);

            double rab_norm_sqr_1 = rab1.x * rab1.x + rab1.y * rab1.y;
            double rabvb_norm_sqr_1 = rabvb1.x * rabvb1.x + rabvb1.y * rabvb1.y;
            double vb_norm_sqr_1 = vb1.x * vb1.x + vb1.y * vb1.y;
            double rab_norm_1 = sqrt(rab_norm_sqr_1);
            double rabvb_norm_1 = sqrt(rabvb_norm_sqr_1);

            double rab_norm_sqr_2 = rab2.x * rab2.x + rab2.y * rab2.y;
            double rabvb_norm_sqr_2 = rabvb2.x * rabvb2.x + rabvb2.y * rabvb2.y;
            double vb_norm_sqr_2 = vb2.x * vb2.x + vb2.y * vb2.y;
            double rab_norm_2 = sqrt(rab_norm_sqr_2);
            double rabvb_norm_2 = sqrt(rabvb_norm_sqr_2);

            double rab_norm_sqr_3 = rab3.x * rab3.x + rab3.y * rab3.y;
            double rabvb_norm_sqr_3 = rabvb3.x * rabvb3.x + rabvb3.y * rabvb3.y;
            double vb_norm_sqr_3 = vb3.x * vb3.x + vb3.y * vb3.y;
            double rab_norm_3 = sqrt(rab_norm_sqr_3);
            double rabvb_norm_3 = sqrt(rabvb_norm_sqr_3);

            norm_sum = rab_norm + rabvb_norm;
            norm_sum1 = rab_norm_1 + rabvb_norm_1;
            norm_sum2 = rab_norm_2 + rabvb_norm_2;
            norm_sum3 = rab_norm_3 + rabvb_norm_3;


            double b = sqrt(norm_sum * norm_sum - vb_norm_sqr) * 0.5; // 4x3+7=19
            double b1 = sqrt(norm_sum1 * norm_sum1 - vb_norm_sqr_1) * 0.5; // 4x3+7=19
            double b2 = sqrt(norm_sum2 * norm_sum2 - vb_norm_sqr_2) * 0.5; // 4x3+7=19
            double b3 = sqrt(norm_sum3 * norm_sum3 - vb_norm_sqr_3) * 0.5; // 4x3+7=19

            double wef; double wef1; double wef2; double wef3;
            Vector2d f_ab(0., 0.);
            if (rab.norm() != 0.) {
                f_ab = _const * exp(-b * sigma_inv) * norm_sum * (rab / rab_norm + rabvb / rabvb_norm) / b;  // 12
            }

            Vector2d f_ab1(0., 0.);
            if (rab1.norm() != 0.) {
                f_ab1 = _const * exp(-b1 * sigma_inv) * norm_sum1 * (rab1 / rab_norm_1 + rabvb1 / rabvb_norm_1) / b1;  // 12
            }

            Vector2d f_ab2(0., 0.);
            if (rab2.norm() != 0.) {
                f_ab2 = _const * exp(-b2 * sigma_inv) * norm_sum2 * (rab2 / rab_norm_2 + rabvb2 / rabvb_norm_2) / b2;  // 12
            }

            Vector2d f_ab3(0., 0.);
            if (rab3.norm() != 0.) {
                f_ab3 = _const * exp(-b3 * sigma_inv) * norm_sum3 * (rab3 / rab_norm_3 + rabvb3 / rabvb_norm_3) / b3;  // 12
            }

            if (e[i].dot(-f_ab) >= f_ab.norm() * cos_phi)
                wef = 1.;
            else
                wef = c;

            if (e[i].dot(-f_ab1) >= f_ab1.norm() * cos_phi)
                wef1 = 1.;
            else
                wef1 = c;

            if (e[i].dot(-f_ab2) >= f_ab2.norm() * cos_phi)
                wef2 = 1.;
            else
                wef2 = c;

            if (e[i].dot(-f_ab3) >= f_ab3.norm() * cos_phi)
                wef3 = 1.;
            else
                wef3 = c;

            F_rep_ped += wef * f_ab + wef1 * f_ab1 + wef2 * f_ab2 + wef3 * f_ab3;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = _const1 * exp(-raB0.norm() / R) * raB0 / raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y - height);
        Vector2d F_aB1 = _const1 * exp(-raB1.norm() / R) * raB1 / raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i] * dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha * dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }
}


// more pre computation than 3.4
// still slower than 3_4
// L1 cache traffic sloweer than just do computation
void optimized_3_4_1(int np, int num_runs) {
    Vector2d vbs[n_p];
    double vb_norm_sqrs[n_p];
#pragma ivdep
    for (int a = 0; a < n_p; a++) {
        vbs[a] = v[a] * Dt;
        vb_norm_sqrs[a] = vbs[a].x * vbs[a].x + vbs[a].y * vbs[a].y;
    }
    double cos_phi = cos(100. / 180 * M_PI);
    double tau_a_inv = 1 / tau_a;
    double sigma_inv = 1 / sigma;
    double norm_sum = 0;
    double _const = vab0 * sigma_inv * 0.25;
    double _const1 = UaB0 / R;
    double flag0 = 1; double flag1 = 1; double flag2 = 1; double flag3 = 1;
#pragma ivdep
    for (int i = 0; i < n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);

        // acceleration term
        Vector2d F0_alpha = (e[i] - v[i]) / tau_a * v0[i]; // 6 flops


        F_alpha += F0_alpha; // 2 flops


        Vector2d F_rep_ped(0., 0.);
        double norm_sum; double norm_sum1; double norm_sum2; double norm_sum3;

#pragma ivdep
        for (int j = 0; j < n_p; j += 4) {

            Vector2d rab = r[i] - r[j];
            Vector2d rab1 = r[i] - r[j + 1];
            Vector2d rab2 = r[i] - r[j + 2];
            Vector2d rab3 = r[i] - r[j + 3];

            Vector2d vb = vbs[j];
            Vector2d vb1 = vbs[j + 1];
            Vector2d vb2 = vbs[j + 2];
            Vector2d vb3 = vbs[j + 3];

            Vector2d rabvb = rab - vb;
            Vector2d rabvb1 = rab1 - vb1;
            Vector2d rabvb2 = rab2 - vb2;
            Vector2d rabvb3 = rab3 - vb3;


            double rab_norm_sqr = rab.x * rab.x + rab.y * rab.y;
            double rabvb_norm_sqr = rabvb.x * rabvb.x + rabvb.y * rabvb.y;
            double vb_norm_sqr = vb_norm_sqrs[j];
            double rab_norm = sqrt(rab_norm_sqr);
            double rabvb_norm = sqrt(rabvb_norm_sqr);

            double rab_norm_sqr_1 = rab1.x * rab1.x + rab1.y * rab1.y;
            double rabvb_norm_sqr_1 = rabvb1.x * rabvb1.x + rabvb1.y * rabvb1.y;
            double vb_norm_sqr_1 = vb_norm_sqrs[j + 1];
            double rab_norm_1 = sqrt(rab_norm_sqr_1);
            double rabvb_norm_1 = sqrt(rabvb_norm_sqr_1);

            double rab_norm_sqr_2 = rab2.x * rab2.x + rab2.y * rab2.y;
            double rabvb_norm_sqr_2 = rabvb2.x * rabvb2.x + rabvb2.y * rabvb2.y;
            double vb_norm_sqr_2 = vb_norm_sqrs[j + 2];
            double rab_norm_2 = sqrt(rab_norm_sqr_2);
            double rabvb_norm_2 = sqrt(rabvb_norm_sqr_2);

            double rab_norm_sqr_3 = rab3.x * rab3.x + rab3.y * rab3.y;
            double rabvb_norm_sqr_3 = rabvb3.x * rabvb3.x + rabvb3.y * rabvb3.y;
            double vb_norm_sqr_3 = vb_norm_sqrs[j + 3];
            double rab_norm_3 = sqrt(rab_norm_sqr_3);
            double rabvb_norm_3 = sqrt(rabvb_norm_sqr_3);

            norm_sum = rab_norm + rabvb_norm;
            norm_sum1 = rab_norm_1 + rabvb_norm_1;
            norm_sum2 = rab_norm_2 + rabvb_norm_2;
            norm_sum3 = rab_norm_3 + rabvb_norm_3;


            double b = sqrt(norm_sum * norm_sum - vb_norm_sqr) * 0.5; // 4x3+7=19
            double b1 = sqrt(norm_sum1 * norm_sum1 - vb_norm_sqr_1) * 0.5; // 4x3+7=19
            double b2 = sqrt(norm_sum2 * norm_sum2 - vb_norm_sqr_2) * 0.5; // 4x3+7=19
            double b3 = sqrt(norm_sum3 * norm_sum3 - vb_norm_sqr_3) * 0.5; // 4x3+7=19

            double wef; double wef1; double wef2; double wef3;
            Vector2d f_ab(0., 0.);
            if (rab.norm() != 0.) {
                f_ab = _const * exp(-b * sigma_inv) * norm_sum * (rab / rab_norm + rabvb / rabvb_norm) / b;  // 12
            }

            Vector2d f_ab1(0., 0.);
            if (rab1.norm() != 0.) {
                f_ab1 = _const * exp(-b1 * sigma_inv) * norm_sum1 * (rab1 / rab_norm_1 + rabvb1 / rabvb_norm_1) / b1;  // 12
            }

            Vector2d f_ab2(0., 0.);
            if (rab2.norm() != 0.) {
                f_ab2 = _const * exp(-b2 * sigma_inv) * norm_sum2 * (rab2 / rab_norm_2 + rabvb2 / rabvb_norm_2) / b2;  // 12
            }

            Vector2d f_ab3(0., 0.);
            if (rab3.norm() != 0.) {
                f_ab3 = _const * exp(-b3 * sigma_inv) * norm_sum3 * (rab3 / rab_norm_3 + rabvb3 / rabvb_norm_3) / b3;  // 12
            }

            if (e[i].dot(-f_ab) >= f_ab.norm() * cos_phi)
                wef = 1.;
            else
                wef = c;

            if (e[i].dot(-f_ab1) >= f_ab1.norm() * cos_phi)
                wef1 = 1.;
            else
                wef1 = c;

            if (e[i].dot(-f_ab2) >= f_ab2.norm() * cos_phi)
                wef2 = 1.;
            else
                wef2 = c;

            if (e[i].dot(-f_ab3) >= f_ab3.norm() * cos_phi)
                wef3 = 1.;
            else
                wef3 = c;

            F_rep_ped += wef * f_ab + wef1 * f_ab1 + wef2 * f_ab2 + wef3 * f_ab3;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = _const1 * exp(-raB0.norm() / R) * raB0 / raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y - height);
        Vector2d F_aB1 = _const1 * exp(-raB1.norm() / R) * raB1 / raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i] * dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha * dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }
}

// unroll outer loop (blocking)
void optimized_3_4_2(int np, int num_runs) {
    Vector2d vbs[n_p];
    double vb_norm_sqrs[n_p];
#pragma ivdep
    for (int a = 0; a < n_p; a++) {
        vbs[a] = v[a] * Dt;
        vb_norm_sqrs[a] = vbs[a].x * vbs[a].x + vbs[a].y * vbs[a].y;
    }
    double cos_phi = cos(100. / 180 * M_PI);
    double tau_a_inv = 1 / tau_a;
    double sigma_inv = 1 / sigma;
    double norm_sum = 0;
    double _const = vab0 * sigma_inv * 0.25;
    double _const1 = UaB0 / R;
    double flag0 = 1; double flag1 = 1; double flag2 = 1; double flag3 = 1;
#pragma ivdep
    for (int i = 0; i < n_p; i += 4) {

        Vector2d F_alpha_i = (e[i] - v[i]) / tau_a * v0[i];
        Vector2d F_alpha_i1 = (e[i + 1] - v[i + 1]) / tau_a * v0[i + 1];
        Vector2d F_alpha_i2 = (e[i + 2] - v[i + 2]) / tau_a * v0[i + 2];
        Vector2d F_alpha_i3 = (e[i + 3] - v[i + 3]) / tau_a * v0[i + 3];

        Vector2d F_rep_i(0., 0.), F_rep_i1(0., 0.), F_rep_i2(0., 0.), F_rep_i3(0., 0.);
        // double norm_sum; double norm_sum1; double norm_sum2; double norm_sum3;

#pragma ivdep
        for (int j = 0; j < n_p; j += 4) {

            Vector2d rabij = r[i] - r[j];
            Vector2d rabij1 = r[i] - r[j + 1];
            Vector2d rabij2 = r[i] - r[j + 2];
            Vector2d rabij3 = r[i] - r[j + 3];
            Vector2d rabi1j = r[i + 1] - r[j];
            Vector2d rabi1j1 = r[i + 1] - r[j + 1];
            Vector2d rabi1j2 = r[i + 1] - r[j + 2];
            Vector2d rabi1j3 = r[i + 1] - r[j + 3];
            Vector2d rabi2j = r[i + 2] - r[j];
            Vector2d rabi2j1 = r[i + 2] - r[j + 1];
            Vector2d rabi2j2 = r[i + 2] - r[j + 2];
            Vector2d rabi2j3 = r[i + 2] - r[j + 3];
            Vector2d rabi3j = r[i + 3] - r[j];
            Vector2d rabi3j1 = r[i + 3] - r[j + 1];
            Vector2d rabi3j2 = r[i + 3] - r[j + 2];
            Vector2d rabi3j3 = r[i + 3] - r[j + 3];

            Vector2d vb = vbs[j];
            Vector2d vb1 = vbs[j + 1];
            Vector2d vb2 = vbs[j + 2];
            Vector2d vb3 = vbs[j + 3];

            Vector2d rabvbij = rabij - vb;
            Vector2d rabvbij1 = rabij1 - vb1;
            Vector2d rabvbij2 = rabij2 - vb2;
            Vector2d rabvbij3 = rabij3 - vb3;
            Vector2d rabvbi1j = rabi1j - vb;
            Vector2d rabvbi1j1 = rabi1j1 - vb1;
            Vector2d rabvbi1j2 = rabi1j2 - vb2;
            Vector2d rabvbi1j3 = rabi1j3 - vb3;
            Vector2d rabvbi2j = rabi2j - vb;
            Vector2d rabvbi2j1 = rabi2j1 - vb1;
            Vector2d rabvbi2j2 = rabi2j2 - vb2;
            Vector2d rabvbi2j3 = rabi2j3 - vb3;
            Vector2d rabvbi3j = rabi3j - vb;
            Vector2d rabvbi3j1 = rabi3j1 - vb1;
            Vector2d rabvbi3j2 = rabi3j2 - vb2;
            Vector2d rabvbi3j3 = rabi3j3 - vb3;

            double rab_norm_sqr_ij = rabij.x * rabij.x + rabij.y * rabij.y;
            double rabvb_norm_sqr_ij = rabvbij.x * rabvbij.x + rabvbij.y * rabvbij.y;
            double rab_norm_sqr_i1j = rabi1j.x * rabi1j.x + rabi1j.y * rabi1j.y;
            double rabvb_norm_sqr_i1j = rabvbi1j.x * rabvbi1j.x + rabvbi1j.y * rabvbi1j.y;
            double rab_norm_sqr_i2j = rabi2j.x * rabi2j.x + rabi2j.y * rabi2j.y;
            double rabvb_norm_sqr_i2j = rabvbi2j.x * rabvbi2j.x + rabvbi2j.y * rabvbi2j.y;
            double rab_norm_sqr_i3j = rabi3j.x * rabi3j.x + rabi3j.y * rabi3j.y;
            double rabvb_norm_sqr_i3j = rabvbi3j.x * rabvbi3j.x + rabvbi3j.y * rabvbi3j.y;
            double vb_norm_sqr_j = vb_norm_sqrs[j];
            double rab_norm_ij = sqrt(rab_norm_sqr_ij);
            double rabvb_norm_ij = sqrt(rabvb_norm_sqr_ij);
            double rab_norm_i1j = sqrt(rab_norm_sqr_i1j);
            double rabvb_norm_i1j = sqrt(rabvb_norm_sqr_i1j);
            double rab_norm_i2j = sqrt(rab_norm_sqr_i2j);
            double rabvb_norm_i2j = sqrt(rabvb_norm_sqr_i2j);
            double rab_norm_i3j = sqrt(rab_norm_sqr_i3j);
            double rabvb_norm_i3j = sqrt(rabvb_norm_sqr_i3j);

            double rab_norm_sqr_ij1 = rabij1.x * rabij1.x + rabij1.y * rabij1.y;
            double rabvb_norm_sqr_ij1 = rabvbij1.x * rabvbij1.x + rabvbij1.y * rabvbij1.y;
            double rab_norm_sqr_i1j1 = rabi1j1.x * rabi1j1.x + rabi1j1.y * rabi1j1.y;
            double rabvb_norm_sqr_i1j1 = rabvbi1j1.x * rabvbi1j1.x + rabvbi1j1.y * rabvbi1j1.y;
            double rab_norm_sqr_i2j1 = rabi2j1.x * rabi2j1.x + rabi2j1.y * rabi2j1.y;
            double rabvb_norm_sqr_i2j1 = rabvbi2j1.x * rabvbi2j1.x + rabvbi2j1.y * rabvbi2j1.y;
            double rab_norm_sqr_i3j1 = rabi3j1.x * rabi3j1.x + rabi3j1.y * rabi3j1.y;
            double rabvb_norm_sqr_i3j1 = rabvbi3j1.x * rabvbi3j1.x + rabvbi3j1.y * rabvbi3j1.y;
            double vb_norm_sqr_j1 = vb_norm_sqrs[j + 1];
            double rab_norm_ij1 = sqrt(rab_norm_sqr_ij1);
            double rabvb_norm_ij1 = sqrt(rabvb_norm_sqr_ij1);
            double rab_norm_i1j1 = sqrt(rab_norm_sqr_i1j1);
            double rabvb_norm_i1j1 = sqrt(rabvb_norm_sqr_i1j1);
            double rab_norm_i2j1 = sqrt(rab_norm_sqr_i2j1);
            double rabvb_norm_i2j1 = sqrt(rabvb_norm_sqr_i2j1);
            double rab_norm_i3j1 = sqrt(rab_norm_sqr_i3j1);
            double rabvb_norm_i3j1 = sqrt(rabvb_norm_sqr_i3j1);

            double rab_norm_sqr_ij2 = rabij2.x * rabij2.x + rabij2.y * rabij2.y;
            double rabvb_norm_sqr_ij2 = rabvbij2.x * rabvbij2.x + rabvbij2.y * rabvbij2.y;
            double rab_norm_sqr_i1j2 = rabi1j2.x * rabi1j2.x + rabi1j2.y * rabi1j2.y;
            double rabvb_norm_sqr_i1j2 = rabvbi1j2.x * rabvbi1j2.x + rabvbi1j2.y * rabvbi1j2.y;
            double rab_norm_sqr_i2j2 = rabi2j2.x * rabi2j2.x + rabi2j2.y * rabi2j2.y;
            double rabvb_norm_sqr_i2j2 = rabvbi2j2.x * rabvbi2j2.x + rabvbi2j2.y * rabvbi2j2.y;
            double rab_norm_sqr_i3j2 = rabi3j2.x * rabi3j2.x + rabi3j2.y * rabi3j2.y;
            double rabvb_norm_sqr_i3j2 = rabvbi3j2.x * rabvbi3j2.x + rabvbi3j2.y * rabvbi3j2.y;
            double vb_norm_sqr_j2 = vb_norm_sqrs[j + 2];
            double rab_norm_ij2 = sqrt(rab_norm_sqr_ij2);
            double rabvb_norm_ij2 = sqrt(rabvb_norm_sqr_ij2);
            double rab_norm_i1j2 = sqrt(rab_norm_sqr_i1j2);
            double rabvb_norm_i1j2 = sqrt(rabvb_norm_sqr_i1j2);
            double rab_norm_i2j2 = sqrt(rab_norm_sqr_i2j2);
            double rabvb_norm_i2j2 = sqrt(rabvb_norm_sqr_i2j2);
            double rab_norm_i3j2 = sqrt(rab_norm_sqr_i3j2);
            double rabvb_norm_i3j2 = sqrt(rabvb_norm_sqr_i3j2);

            double rab_norm_sqr_ij3 = rabij3.x * rabij3.x + rabij3.y * rabij3.y;
            double rabvb_norm_sqr_ij3 = rabvbij3.x * rabvbij3.x + rabvbij3.y * rabvbij3.y;
            double rab_norm_sqr_i1j3 = rabi1j3.x * rabi1j3.x + rabi1j3.y * rabi1j3.y;
            double rabvb_norm_sqr_i1j3 = rabvbi1j3.x * rabvbi1j3.x + rabvbi1j3.y * rabvbi1j3.y;
            double rab_norm_sqr_i2j3 = rabi2j3.x * rabi2j3.x + rabi2j3.y * rabi2j3.y;
            double rabvb_norm_sqr_i2j3 = rabvbi2j3.x * rabvbi2j3.x + rabvbi2j3.y * rabvbi2j3.y;
            double rab_norm_sqr_i3j3 = rabi3j3.x * rabi3j3.x + rabi3j3.y * rabi3j3.y;
            double rabvb_norm_sqr_i3j3 = rabvbi3j3.x * rabvbi3j3.x + rabvbi3j3.y * rabvbi3j3.y;
            double vb_norm_sqr_j3 = vb_norm_sqrs[j + 3];
            double rab_norm_ij3 = sqrt(rab_norm_sqr_ij3);
            double rabvb_norm_ij3 = sqrt(rabvb_norm_sqr_ij3);
            double rab_norm_i1j3 = sqrt(rab_norm_sqr_i1j3);
            double rabvb_norm_i1j3 = sqrt(rabvb_norm_sqr_i1j3);
            double rab_norm_i2j3 = sqrt(rab_norm_sqr_i2j3);
            double rabvb_norm_i2j3 = sqrt(rabvb_norm_sqr_i2j3);
            double rab_norm_i3j3 = sqrt(rab_norm_sqr_i3j3);
            double rabvb_norm_i3j3 = sqrt(rabvb_norm_sqr_i3j3);

            double norm_sumij = rab_norm_ij + rabvb_norm_ij;
            double norm_sumij1 = rab_norm_ij1 + rabvb_norm_ij1;
            double norm_sumij2 = rab_norm_ij2 + rabvb_norm_ij2;
            double norm_sumij3 = rab_norm_ij3 + rabvb_norm_ij3;
            double norm_sumi1j = rab_norm_i1j + rabvb_norm_i1j;
            double norm_sumi1j1 = rab_norm_i1j1 + rabvb_norm_i1j1;
            double norm_sumi1j2 = rab_norm_i1j2 + rabvb_norm_i1j2;
            double norm_sumi1j3 = rab_norm_i1j3 + rabvb_norm_i1j3;
            double norm_sumi2j = rab_norm_i2j + rabvb_norm_i2j;
            double norm_sumi2j1 = rab_norm_i2j1 + rabvb_norm_i2j1;
            double norm_sumi2j2 = rab_norm_i2j2 + rabvb_norm_i2j2;
            double norm_sumi2j3 = rab_norm_i2j3 + rabvb_norm_i2j3;
            double norm_sumi3j = rab_norm_i3j + rabvb_norm_i3j;
            double norm_sumi3j1 = rab_norm_i3j1 + rabvb_norm_i3j1;
            double norm_sumi3j2 = rab_norm_i3j2 + rabvb_norm_i3j2;
            double norm_sumi3j3 = rab_norm_i3j3 + rabvb_norm_i3j3;

            double bij = sqrt(norm_sumij * norm_sumij - vb_norm_sqr_j) * 0.5;
            double bij1 = sqrt(norm_sumij1 * norm_sumij1 - vb_norm_sqr_j1) * 0.5;
            double bij2 = sqrt(norm_sumij2 * norm_sumij2 - vb_norm_sqr_j2) * 0.5;
            double bij3 = sqrt(norm_sumij3 * norm_sumij3 - vb_norm_sqr_j3) * 0.5;
            double bi1j = sqrt(norm_sumi1j * norm_sumi1j - vb_norm_sqr_j) * 0.5;
            double bi1j1 = sqrt(norm_sumi1j1 * norm_sumi1j1 - vb_norm_sqr_j1) * 0.5;
            double bi1j2 = sqrt(norm_sumi1j2 * norm_sumi1j2 - vb_norm_sqr_j2) * 0.5;
            double bi1j3 = sqrt(norm_sumi1j3 * norm_sumi1j3 - vb_norm_sqr_j3) * 0.5;
            double bi2j = sqrt(norm_sumi2j * norm_sumi2j - vb_norm_sqr_j) * 0.5;
            double bi2j1 = sqrt(norm_sumi2j1 * norm_sumi2j1 - vb_norm_sqr_j1) * 0.5;
            double bi2j2 = sqrt(norm_sumi2j2 * norm_sumi2j2 - vb_norm_sqr_j2) * 0.5;
            double bi2j3 = sqrt(norm_sumi2j3 * norm_sumi2j3 - vb_norm_sqr_j3) * 0.5;
            double bi3j = sqrt(norm_sumi3j * norm_sumi3j - vb_norm_sqr_j) * 0.5;
            double bi3j1 = sqrt(norm_sumi3j1 * norm_sumi3j1 - vb_norm_sqr_j1) * 0.5;
            double bi3j2 = sqrt(norm_sumi3j2 * norm_sumi3j2 - vb_norm_sqr_j2) * 0.5;
            double bi3j3 = sqrt(norm_sumi3j3 * norm_sumi3j3 - vb_norm_sqr_j3) * 0.5;

            double wefij, wefij1, wefij2, wefij3, wefi1j, wefi1j1, wefi1j2, wefi1j3, wefi2j, wefi2j1, wefi2j2, wefi2j3, wefi3j, wefi3j1, wefi3j2, wefi3j3;
            Vector2d f_abij(0., 0.);
            if (rabij.norm() != 0.) {
                f_abij = _const * exp(-bij * sigma_inv) * norm_sumij * (rabij / rab_norm_ij + rabvbij / rabvb_norm_ij) / bij;  // 12
            }
            Vector2d f_abij1(0., 0.);
            if (rabij1.norm() != 0.) {
                f_abij1 = _const * exp(-bij1 * sigma_inv) * norm_sumij1 * (rabij1 / rab_norm_ij1 + rabvbij1 / rabvb_norm_ij1) / bij1;  // 12
            }
            Vector2d f_abij2(0., 0.);
            if (rabij2.norm() != 0.) {
                f_abij2 = _const * exp(-bij2 * sigma_inv) * norm_sumij2 * (rabij2 / rab_norm_ij2 + rabvbij2 / rabvb_norm_ij2) / bij2;  // 12
            }
            Vector2d f_abij3(0., 0.);
            if (rabij3.norm() != 0.) {
                f_abij3 = _const * exp(-bij3 * sigma_inv) * norm_sumij3 * (rabij3 / rab_norm_ij3 + rabvbij3 / rabvb_norm_ij3) / bij3;  // 12
            }

            Vector2d f_abi1j(0., 0.);
            if (rabi1j.norm() != 0.) {
                f_abi1j = _const * exp(-bi1j * sigma_inv) * norm_sumi1j * (rabi1j / rab_norm_i1j + rabvbi1j / rabvb_norm_i1j) / bi1j;  // 12
            }
            Vector2d f_abi1j1(0., 0.);
            if (rabi1j1.norm() != 0.) {
                f_abi1j1 = _const * exp(-bi1j1 * sigma_inv) * norm_sumi1j1 * (rabi1j1 / rab_norm_i1j1 + rabvbi1j1 / rabvb_norm_i1j1) / bi1j1;  // 12
            }
            Vector2d f_abi1j2(0., 0.);
            if (rabi1j2.norm() != 0.) {
                f_abi1j2 = _const * exp(-bi1j2 * sigma_inv) * norm_sumi1j2 * (rabi1j2 / rab_norm_i1j2 + rabvbi1j2 / rabvb_norm_i1j2) / bi1j2;  // 12
            }
            Vector2d f_abi1j3(0., 0.);
            if (rabi1j3.norm() != 0.) {
                f_abi1j3 = _const * exp(-bi1j3 * sigma_inv) * norm_sumi1j3 * (rabi1j3 / rab_norm_i1j3 + rabvbi1j3 / rabvb_norm_i1j3) / bi1j3;  // 12
            }

            Vector2d f_abi2j(0., 0.);
            if (rabi2j.norm() != 0.) {
                f_abi2j = _const * exp(-bi2j * sigma_inv) * norm_sumi2j * (rabi2j / rab_norm_i2j + rabvbi2j / rabvb_norm_i2j) / bi2j;  // 12
            }
            Vector2d f_abi2j1(0., 0.);
            if (rabi2j1.norm() != 0.) {
                f_abi2j1 = _const * exp(-bi2j1 * sigma_inv) * norm_sumi2j1 * (rabi2j1 / rab_norm_i2j1 + rabvbi2j1 / rabvb_norm_i2j1) / bi2j1;  // 12
            }
            Vector2d f_abi2j2(0., 0.);
            if (rabi2j2.norm() != 0.) {
                f_abi2j2 = _const * exp(-bi2j2 * sigma_inv) * norm_sumi2j2 * (rabi2j2 / rab_norm_i2j2 + rabvbi2j2 / rabvb_norm_i2j2) / bi2j2;  // 12
            }
            Vector2d f_abi2j3(0., 0.);
            if (rabi2j3.norm() != 0.) {
                f_abi2j3 = _const * exp(-bi2j3 * sigma_inv) * norm_sumi2j3 * (rabi2j3 / rab_norm_i2j3 + rabvbi2j3 / rabvb_norm_i2j3) / bi2j3;  // 12
            }

            Vector2d f_abi3j(0., 0.);
            if (rabi3j.norm() != 0.) {
                f_abi3j = _const * exp(-bi3j * sigma_inv) * norm_sumi3j * (rabi3j / rab_norm_i3j + rabvbi3j / rabvb_norm_i3j) / bi3j;  // 12
            }
            Vector2d f_abi3j1(0., 0.);
            if (rabi3j1.norm() != 0.) {
                f_abi3j1 = _const * exp(-bi3j1 * sigma_inv) * norm_sumi3j1 * (rabi3j1 / rab_norm_i3j1 + rabvbi3j1 / rabvb_norm_i3j1) / bi3j1;  // 12
            }
            Vector2d f_abi3j2(0., 0.);
            if (rabi3j2.norm() != 0.) {
                f_abi3j2 = _const * exp(-bi3j2 * sigma_inv) * norm_sumi3j2 * (rabi3j2 / rab_norm_i3j2 + rabvbi3j2 / rabvb_norm_i3j2) / bi3j2;  // 12
            }
            Vector2d f_abi3j3(0., 0.);
            if (rabi3j3.norm() != 0.) {
                f_abi3j3 = _const * exp(-bi3j3 * sigma_inv) * norm_sumi3j3 * (rabi3j3 / rab_norm_i3j3 + rabvbi3j3 / rabvb_norm_i3j3) / bi3j3;  // 12
            }

            if (e[i].dot(-f_abij) >= f_abij.norm() * cos_phi)
                wefij = 1.;
            else
                wefij = c;
            if (e[i].dot(-f_abij1) >= f_abij1.norm() * cos_phi)
                wefij1 = 1.;
            else
                wefij1 = c;
            if (e[i].dot(-f_abij2) >= f_abij2.norm() * cos_phi)
                wefij2 = 1.;
            else
                wefij2 = c;
            if (e[i].dot(-f_abij3) >= f_abij3.norm() * cos_phi)
                wefij3 = 1.;
            else
                wefij3 = c;

            if (e[i + 1].dot(-f_abi1j) >= f_abi1j.norm() * cos_phi)
                wefi1j = 1.;
            else
                wefi1j = c;
            if (e[i + 1].dot(-f_abi1j1) >= f_abi1j1.norm() * cos_phi)
                wefi1j1 = 1.;
            else
                wefi1j1 = c;
            if (e[i + 1].dot(-f_abi1j2) >= f_abi1j2.norm() * cos_phi)
                wefi1j2 = 1.;
            else
                wefi1j2 = c;
            if (e[i + 1].dot(-f_abi1j3) >= f_abi1j3.norm() * cos_phi)
                wefi1j3 = 1.;
            else
                wefi1j3 = c;

            if (e[i + 2].dot(-f_abi2j) >= f_abi2j.norm() * cos_phi)
                wefi2j = 1.;
            else
                wefi2j = c;
            if (e[i + 2].dot(-f_abi2j1) >= f_abi2j1.norm() * cos_phi)
                wefi2j1 = 1.;
            else
                wefi2j1 = c;
            if (e[i + 2].dot(-f_abi2j2) >= f_abi2j2.norm() * cos_phi)
                wefi2j2 = 1.;
            else
                wefi2j2 = c;
            if (e[i + 2].dot(-f_abi2j3) >= f_abi2j3.norm() * cos_phi)
                wefi2j3 = 1.;
            else
                wefi2j3 = c;

            if (e[i + 3].dot(-f_abi3j) >= f_abi3j.norm() * cos_phi)
                wefi3j = 1.;
            else
                wefi3j = c;
            if (e[i + 3].dot(-f_abi3j1) >= f_abi3j1.norm() * cos_phi)
                wefi3j1 = 1.;
            else
                wefi3j1 = c;
            if (e[i + 3].dot(-f_abi3j2) >= f_abi3j2.norm() * cos_phi)
                wefi3j2 = 1.;
            else
                wefi3j2 = c;
            if (e[i + 3].dot(-f_abi3j3) >= f_abi3j3.norm() * cos_phi)
                wefi3j3 = 1.;
            else
                wefi3j3 = c;

            F_rep_i += wefij * f_abij + wefij1 * f_abij1 + wefij2 * f_abij2 + wefij3 * f_abij3;
            F_rep_i1 += wefi1j * f_abi1j + wefi1j1 * f_abi1j1 + wefi1j2 * f_abi1j2 + wefi1j3 * f_abi1j3;
            F_rep_i2 += wefi2j * f_abi2j + wefi2j1 * f_abi2j1 + wefi2j2 * f_abi2j2 + wefi2j3 * f_abi2j3;
            F_rep_i3 += wefi3j * f_abi3j + wefi3j1 * f_abi3j1 + wefi3j2 * f_abi3j2 + wefi3j3 * f_abi3j3;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha_i += F_rep_i;
        F_alpha_i1 += F_rep_i1;
        F_alpha_i2 += F_rep_i2;
        F_alpha_i3 += F_rep_i3;

        Vector2d raB0i(0., r[i].y);
        Vector2d F_aB0i = _const1 * exp(-raB0i.norm() / R) * raB0i / raB0i.norm(); // 15 flops
        Vector2d raB1i(0., r[i].y - height);
        Vector2d F_aB1i = _const1 * exp(-raB1i.norm() / R) * raB1i / raB1i.norm();
        F_alpha_i += F_aB0i + F_aB1i;
        Vector2d raB0i1(0., r[i + 1].y);
        Vector2d F_aB0i1 = _const1 * exp(-raB0i1.norm() / R) * raB0i1 / raB0i1.norm();
        Vector2d raB1i1(0., r[i + 1].y - height);
        Vector2d F_aB1i1 = _const1 * exp(-raB1i1.norm() / R) * raB1i1 / raB1i1.norm();
        F_alpha_i1 += F_aB0i1 + F_aB1i1;
        Vector2d raB0i2(0., r[i + 2].y);
        Vector2d F_aB0i2 = _const1 * exp(-raB0i2.norm() / R) * raB0i2 / raB0i2.norm();
        Vector2d raB1i2(0., r[i + 2].y - height);
        Vector2d F_aB1i2 = _const1 * exp(-raB1i2.norm() / R) * raB1i2 / raB1i2.norm();
        F_alpha_i2 += F_aB0i2 + F_aB1i2;
        Vector2d raB0i3(0., r[i + 3].y);
        Vector2d F_aB0i3 = _const1 * exp(-raB0i3.norm() / R) * raB0i3 / raB0i3.norm();
        Vector2d raB1i3(0., r[i + 3].y - height);
        Vector2d F_aB1i3 = _const1 * exp(-raB1i3.norm() / R) * raB1i3 / raB1i3.norm();
        F_alpha_i3 += F_aB0i3 + F_aB1i3;

        r_next[i] += v[i] * dt;
        Vector2d w_alpha_i = v[i] + F_alpha_i * dt;
        v_next[i] = w_alpha_i * g(vMax[i], w_alpha_i.norm());
        r_next[i + 1] += v[i + 1] * dt;
        Vector2d w_alpha_i1 = v[i + 1] + F_alpha_i1 * dt;
        v_next[i + 1] = w_alpha_i1 * g(vMax[i + 1], w_alpha_i1.norm());
        r_next[i + 2] += v[i + 2] * dt;
        Vector2d w_alpha_i2 = v[i + 2] + F_alpha_i2 * dt;
        v_next[i + 2] = w_alpha_i2 * g(vMax[i + 2], w_alpha_i2.norm());
        r_next[i + 3] += v[i + 3] * dt;
        Vector2d w_alpha_i3 = v[i + 3] + F_alpha_i3 * dt;
        v_next[i + 3] = w_alpha_i3 * g(vMax[i + 3], w_alpha_i3.norm());
    }
}




// 3.5. unroll loops + prev optimizations
// slower than 3--- compiler doesnt like this, compiler avxed 3 so its very fast
void optimized_3_5(int np, int num_runs) {
    double cos_phi = cos(100. / 180 * M_PI);
    double tau_a_inv = 1 / tau_a;
    double sigma_inv = 1 / sigma;
    double norm_sum = 0;
    double _const = vab0 * sigma_inv * 0.25;
    double _const1 = UaB0 / R;

    double flag0 = 1; double flag1 = 1; double flag2 = 1; double flag3 = 1;
    for (int i = 0; i < n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);

        // acceleration term
        Vector2d F0_alpha = (e[i] - v[i]) * tau_a_inv * v0[i]; // 6 flops


        F_alpha += F0_alpha; // 2 flops


        Vector2d F_rep_ped(0., 0.);
        double norm_sum; double norm_sum1; double norm_sum2; double norm_sum3;

        for (int j = 0; j < n_p; j += 4) {

            Vector2d rab = r[i] - r[j];
            Vector2d rab1 = r[i] - r[j + 1];
            Vector2d rab2 = r[i] - r[j + 2];
            Vector2d rab3 = r[i] - r[j + 3];

            Vector2d vb = v[j] * Dt;
            Vector2d vb1 = v[j + 1] * Dt;
            Vector2d vb2 = v[j + 2] * Dt;
            Vector2d vb3 = v[j + 3] * Dt;

            Vector2d rabvb = rab - vb;
            Vector2d rabvb1 = rab1 - vb1;
            Vector2d rabvb2 = rab2 - vb2;
            Vector2d rabvb3 = rab3 - vb3;


            double rab_norm_sqr = rab.x * rab.x + rab.y * rab.y;
            double rabvb_norm_sqr = rabvb.x * rabvb.x + rabvb.y * rabvb.y;
            double vb_norm_sqr = vb.x * vb.x + vb.y * vb.y;
            double rab_norm = sqrt(rab_norm_sqr);
            double rabvb_norm = sqrt(rabvb_norm_sqr);

            double rab_norm_sqr_1 = rab1.x * rab1.x + rab1.y * rab1.y;
            double rabvb_norm_sqr_1 = rabvb1.x * rabvb1.x + rabvb1.y * rabvb1.y;
            double vb_norm_sqr_1 = vb1.x * vb1.x + vb1.y * vb1.y;
            double rab_norm_1 = sqrt(rab_norm_sqr_1);
            double rabvb_norm_1 = sqrt(rabvb_norm_sqr_1);

            double rab_norm_sqr_2 = rab2.x * rab2.x + rab2.y * rab2.y;
            double rabvb_norm_sqr_2 = rabvb2.x * rabvb2.x + rabvb2.y * rabvb2.y;
            double vb_norm_sqr_2 = vb2.x * vb2.x + vb2.y * vb2.y;
            double rab_norm_2 = sqrt(rab_norm_sqr_2);
            double rabvb_norm_2 = sqrt(rabvb_norm_sqr_2);

            double rab_norm_sqr_3 = rab3.x * rab3.x + rab3.y * rab3.y;
            double rabvb_norm_sqr_3 = rabvb3.x * rabvb3.x + rabvb3.y * rabvb3.y;
            double vb_norm_sqr_3 = vb3.x * vb3.x + vb3.y * vb3.y;
            double rab_norm_3 = sqrt(rab_norm_sqr_3);
            double rabvb_norm_3 = sqrt(rabvb_norm_sqr_3);

            norm_sum = rab_norm + rabvb_norm;
            norm_sum1 = rab_norm_1 + rabvb_norm_1;
            norm_sum2 = rab_norm_2 + rabvb_norm_2;
            norm_sum3 = rab_norm_3 + rabvb_norm_3;


            double b = sqrt(norm_sum * norm_sum - vb_norm_sqr) * 0.5; // 4x3+7=19
            double b1 = sqrt(norm_sum1 * norm_sum1 - vb_norm_sqr_1) * 0.5; // 4x3+7=19
            double b2 = sqrt(norm_sum2 * norm_sum2 - vb_norm_sqr_2) * 0.5; // 4x3+7=19
            double b3 = sqrt(norm_sum3 * norm_sum3 - vb_norm_sqr_3) * 0.5; // 4x3+7=19

            double rab_norm_inv = rab_norm_sqr;
            double x2 = rab_norm_inv * 0.5;
            std::int64_t x3 = *(std::int64_t*)&rab_norm_inv;
            x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
            rab_norm_inv = *(double*)&x3;
            rab_norm_inv = rab_norm_inv * (1.5 - (x2 * rab_norm_inv * rab_norm_inv));

            double rab_norm_inv_1 = rab_norm_sqr_1;
            x2 = rab_norm_inv_1 * 0.5;
            x3 = *(std::int64_t*)&rab_norm_inv_1;
            x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
            rab_norm_inv_1 = *(double*)&x3;
            rab_norm_inv_1 = rab_norm_inv_1 * (1.5 - (x2 * rab_norm_inv_1 * rab_norm_inv_1));

            double rab_norm_inv_2 = rab_norm_sqr_2;
            x2 = rab_norm_inv_2 * 0.5;
            x3 = *(std::int64_t*)&rab_norm_inv_2;
            x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
            rab_norm_inv_2 = *(double*)&x3;
            rab_norm_inv_2 = rab_norm_inv_2 * (1.5 - (x2 * rab_norm_inv_2 * rab_norm_inv_2));

            double rab_norm_inv_3 = rab_norm_sqr_3;
            x2 = rab_norm_inv_3 * 0.5;
            x3 = *(std::int64_t*)&rab_norm_inv_3;
            x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
            rab_norm_inv_3 = *(double*)&x3;
            rab_norm_inv_3 = rab_norm_inv_3 * (1.5 - (x2 * rab_norm_inv_3 * rab_norm_inv_3));

            double rabvb_norm_inv = rabvb_norm_sqr;
            x2 = rabvb_norm_inv * 0.5;
            x3 = *(std::int64_t*)&rabvb_norm_inv;
            x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
            rabvb_norm_inv = *(double*)&x3;
            rabvb_norm_inv = rabvb_norm_inv * (1.5 - (x2 * rabvb_norm_inv * rabvb_norm_inv));

            double rabvb_norm_inv_1 = rabvb_norm_sqr_1;
            x2 = rabvb_norm_inv_1 * 0.5;
            x3 = *(std::int64_t*)&rabvb_norm_inv_1;
            x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
            rabvb_norm_inv_1 = *(double*)&x3;
            rabvb_norm_inv_1 = rabvb_norm_inv_1 * (1.5 - (x2 * rabvb_norm_inv_1 * rabvb_norm_inv_1));

            double rabvb_norm_inv_2 = rabvb_norm_sqr_2;
            x2 = rabvb_norm_inv_2 * 0.5;
            x3 = *(std::int64_t*)&rabvb_norm_inv_2;
            x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
            rabvb_norm_inv_2 = *(double*)&x3;
            rabvb_norm_inv_2 = rabvb_norm_inv_2 * (1.5 - (x2 * rabvb_norm_inv_2 * rabvb_norm_inv_2));


            double rabvb_norm_inv_3 = rabvb_norm_sqr_3;
            x2 = rabvb_norm_inv_3 * 0.5;
            x3 = *(std::int64_t*)&rabvb_norm_inv_3;
            x3 = 0x5fe6eb50c7b537a9 - (x3 >> 1);
            rabvb_norm_inv_3 = *(double*)&x3;
            rabvb_norm_inv_3 = rabvb_norm_inv_3 * (1.5 - (x2 * rabvb_norm_inv_3 * rabvb_norm_inv_3));

            double wef; double wef1; double wef2; double wef3;
            Vector2d f_ab(0., 0.);
            if (rab.norm() != 0.) {
                f_ab = _const * exp(-b * sigma_inv) * norm_sum * (rab * rab_norm_inv + rabvb * rabvb_norm_inv) / b;  // 12
            }

            Vector2d f_ab1(0., 0.);
            if (rab1.norm() != 0.) {
                f_ab1 = _const * exp(-b1 * sigma_inv) * norm_sum1 * (rab1 * rab_norm_inv_1 + rabvb1 * rabvb_norm_inv_1) / b1;  // 12
            }

            Vector2d f_ab2(0., 0.);
            if (rab2.norm() != 0.) {
                f_ab2 = _const * exp(-b2 * sigma_inv) * norm_sum2 * (rab2 * rab_norm_inv_2 + rabvb2 * rabvb_norm_inv_2) / b2;  // 12
            }

            Vector2d f_ab3(0., 0.);
            if (rab3.norm() != 0.) {
                f_ab3 = _const * exp(-b3 * sigma_inv) * norm_sum3 * (rab3 * rab_norm_inv_3 + rabvb3 * rabvb_norm_inv_3) / b3;  // 12
            }

            if (e[i].dot(-f_ab) >= -f_ab.norm() * cos_phi)
                wef = 1.;
            else
                wef = c;

            if (e[i].dot(-f_ab1) >= -f_ab1.norm() * cos_phi)
                wef1 = 1.;
            else
                wef1 = c;

            if (e[i].dot(-f_ab2) >= -f_ab2.norm() * cos_phi)
                wef2 = 1.;
            else
                wef2 = c;

            if (e[i].dot(-f_ab3) >= -f_ab3.norm() * cos_phi)
                wef3 = 1.;
            else
                wef3 = c;

            F_rep_ped += wef * f_ab + wef1 * f_ab1 + wef2 * f_ab2 + wef3 * f_ab3;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = _const1 * exp(-raB0.norm() / R) * raB0 / raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y - height);
        Vector2d F_aB1 = _const1 * exp(-raB1.norm() / R) * raB1 / raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i] * dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha * dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }
}


// 3.6 is like 3.4 pre-calculate norm
// still slower than 3.4
void optimized_3_6(int np, int num_runs) {
    double cos_phi = cos(100. / 180 * M_PI);
    double tau_a_inv = 1 / tau_a;
    double sigma_inv = 1 / sigma;
    double norm_sum = 0;
    double _const = vab0 * sigma_inv * 0.25;
    double _const1 = UaB0 / R;
    double flag0 = 1; double flag1 = 1; double flag2 = 1; double flag3 = 1;
    for (int i = 0; i < n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha(.0, .0);

        // acceleration term
        Vector2d F0_alpha = (e[i] - v[i]) / tau_a * v0[i]; // 6 flops


        F_alpha += F0_alpha; // 2 flops


        Vector2d F_rep_ped(0., 0.);
        double norm_sum; double norm_sum1; double norm_sum2; double norm_sum3;

        for (int j = 0; j < n_p; j += 4) {

            Vector2d rab = r[i] - r[j];
            Vector2d rab_1 = r[i] - r[j + 1];
            Vector2d rab_2 = r[i] - r[j + 2];
            Vector2d rab_3 = r[i] - r[j + 3];

            Vector2d vb = v[j] * Dt;
            Vector2d vb_1 = v[j + 1] * Dt;
            Vector2d vb_2 = v[j + 2] * Dt;
            Vector2d vb_3 = v[j + 3] * Dt;

            Vector2d rabvb = rab - vb;
            Vector2d rabvb_1 = rab_1 - vb_1;
            Vector2d rabvb_2 = rab_2 - vb_2;
            Vector2d rabvb_3 = rab_3 - vb_3;


            double rab_norm_sqr = rab.x * rab.x + rab.y * rab.y;
            double rabvb_norm_sqr;
            double vb_norm_sqr;
            double rab_norm;
            double rabvb_norm;

            double rab_norm_sqr_1 = rab_1.x * rab_1.x + rab_1.y * rab_1.y;
            double rabvb_norm_sqr_1;
            double vb_norm_sqr_1;
            double rab_norm_1;
            double rabvb_norm_1;

            double rab_norm_sqr_2 = rab_2.x * rab_2.x + rab_2.y * rab_2.y;
            double rabvb_norm_sqr_2;
            double vb_norm_sqr_2;
            double rab_norm_2;
            double rabvb_norm_2;

            double rab_norm_sqr_3 = rab_3.x * rab_3.x + rab_3.y * rab_3.y;;
            double rabvb_norm_sqr_3;
            double vb_norm_sqr_3;
            double rab_norm_3;
            double rabvb_norm_3;



            double b; // 4x3+7=19
            double b1; // 4x3+7=19
            double b2; // 4x3+7=19
            double b3; // 4x3+7=19

            double wef; double wef1; double wef2; double wef3;
            Vector2d f_ab(0., 0.);
            if (rab_norm_sqr != 0.) {
                rab_norm = sqrt(rab_norm_sqr);
                rabvb_norm_sqr = rabvb.x * rabvb.x + rabvb.y * rabvb.y;
                vb_norm_sqr = vb.x * vb.x + vb.y * vb.y;
                rabvb_norm = sqrt(rabvb_norm_sqr);
                norm_sum = rab_norm + rabvb_norm;
                b = sqrt(norm_sum * norm_sum - vb_norm_sqr) * 0.5;
                f_ab = _const * exp(-b * sigma_inv) * norm_sum * (rab / rab_norm + rabvb / rabvb_norm) / b;  // 12
            }

            Vector2d f_ab1(0., 0.);
            if (rab_norm_sqr_1 != 0.) {
                rab_norm_1 = sqrt(rab_norm_sqr_1);
                rabvb_norm_sqr_1 = rabvb_1.x * rabvb_1.x + rabvb_1.y * rabvb_1.y;
                vb_norm_sqr_1 = vb_1.x * vb_1.x + vb_1.y * vb_1.y;

                rabvb_norm_1 = sqrt(rabvb_norm_sqr_1);
                norm_sum1 = rab_norm_1 + rabvb_norm_1;
                b1 = sqrt(norm_sum1 * norm_sum1 - vb_norm_sqr_1) * 0.5;
                f_ab1 = _const * exp(-b1 * sigma_inv) * norm_sum1 * (rab_1 / rab_norm_1 + rabvb_1 / rabvb_norm_1) / b1;  // 12
            }

            Vector2d f_ab2(0., 0.);
            if (rab_norm_sqr_2 != 0.) {
                rab_norm_2 = sqrt(rab_norm_sqr_2);
                rabvb_norm_sqr_2 = rabvb_2.x * rabvb_2.x + rabvb_2.y * rabvb_2.y;
                vb_norm_sqr_2 = vb_2.x * vb_2.x + vb_2.y * vb_2.y;

                rabvb_norm_2 = sqrt(rabvb_norm_sqr_2);
                norm_sum2 = rab_norm_2 + rabvb_norm_2;
                b2 = sqrt(norm_sum2 * norm_sum2 - vb_norm_sqr_2) * 0.5;
                f_ab2 = _const * exp(-b2 * sigma_inv) * norm_sum2 * (rab_2 / rab_norm_2 + rabvb_2 / rabvb_norm_2) / b2;
            }

            Vector2d f_ab3(0., 0.);
            if (rab_norm_sqr_3 != 0.) {
                rab_norm_3 = sqrt(rab_norm_sqr_3);
                rabvb_norm_sqr_3 = rabvb_3.x * rabvb_3.x + rabvb_3.y * rabvb_3.y;
                vb_norm_sqr_3 = vb_3.x * vb_3.x + vb_3.y * vb_3.y;

                rabvb_norm_3 = sqrt(rabvb_norm_sqr_3);
                norm_sum3 = rab_norm_3 + rabvb_norm_3;
                b3 = sqrt(norm_sum3 * norm_sum3 - vb_norm_sqr_3) * 0.5;
                f_ab3 = _const * exp(-b3 * sigma_inv) * norm_sum3 * (rab_3 / rab_norm_3 + rabvb_3 / rabvb_norm_3) / b3;  // 12
            }

            if (e[i].dot(-f_ab) >= -f_ab.norm() * cos_phi)
                wef = 1.;
            else
                wef = c;

            if (e[i].dot(-f_ab1) >= -f_ab1.norm() * cos_phi)
                wef1 = 1.;
            else
                wef1 = c;

            if (e[i].dot(-f_ab2) >= -f_ab2.norm() * cos_phi)
                wef2 = 1.;
            else
                wef2 = c;

            if (e[i].dot(-f_ab3) >= -f_ab3.norm() * cos_phi)
                wef3 = 1.;
            else
                wef3 = c;

            F_rep_ped += wef * f_ab + wef1 * f_ab1 + wef2 * f_ab2 + wef3 * f_ab3;
            // cout<<rx[i]<<endl;

        }
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha += F_rep_ped; // 2 flops

        Vector2d raB0(0., r[i].y);
        Vector2d F_aB0 = _const1 * exp(-raB0.norm() / R) * raB0 / raB0.norm(); // 15 flops
        Vector2d raB1(0., r[i].y - height);
        Vector2d F_aB1 = _const1 * exp(-raB1.norm() / R) * raB1 / raB1.norm();

        F_alpha += F_aB0 + F_aB1; // 4 flops

        r_next[i] += v[i] * dt; // 4 flops
        Vector2d w_alpha = v[i] + F_alpha * dt; // 4 flops
        v_next[i] = w_alpha * g(vMax[i], w_alpha.norm()); // 4+2+1=7 flops
    }
}


void register_functions() {
    add_function(&baseline, "baseline");
    add_function(&optimized_1, "optimized_1");
    add_function(&optimized_2, "optimized_2");
    add_function(&optimized_2_5, "optimized_2_5");
    add_function(&optimized_3, "optimized_3");
    add_function(&optimized_3_4, "optimized_3_4");
    add_function(&optimized_3_4_1, "optimized_3_4_1");
    add_function(&optimized_3_4_2, "optimized_3_4_2");
    add_function(&optimized_3_5, "optimized_3_5");
    // add_function(&optimized_3_6, "optimized_3_6");

}

void updateFile(string name) {
    std::ofstream file;
    file.open(name);
    update(file);
}

void init_comp(int n_p) {
    init(n_p);
}