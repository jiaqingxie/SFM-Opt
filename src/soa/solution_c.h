#include "social_force_c.h"

void baseline(){
    for (int i=0; i<n_p; i++) {

        // pedestrian *pi = &pedestrians[i];
        Vector2d F_alpha = create_vector(0.0, 0.0);
        
        // acceleration term
        Vector2d F0_alpha = Acceleration_term(i); // 6 flops
        // cout<<"F0_alpha: "<<F0_alpha<<endl;

        F_alpha = vector_add(F_alpha, F0_alpha); // 2 flops

        Vector2d F_ab = Repulsive_pedestrian(i); // 52n flops
        // cout<<"F_ab: "<<F_ab<<endl;
        F_alpha = vector_add(F_alpha, F_ab); // 2 flops

        Vector2d raB0 = create_vector(0.0, r[i].y);
        Vector2d F_aB0 = Repulsive_border(raB0); // 15 flops
        Vector2d raB1 = create_vector(0., r[i].y-height);
        Vector2d F_aB1 = Repulsive_border(raB1);

        F_alpha = vector_add(vector_add(F_aB0, F_aB1), F_alpha); // 4 flops

        r_next[i] = vector_add(r_next[i], vector_mult(v[i], dt)); // 4 flops
        Vector2d w_alpha = vector_add(v[i], vector_mult(F_alpha, dt)); // 4 flops
        v_next[i] = vector_mult(w_alpha , g(vMax[i], vector_norm(w_alpha))); // 4+2+1=7 flops
    }


}


// inline procedure calls
// code motion
// strength reduction
// pair<Vector2d, Vector2d> optimized_1(pedestrian pedestrians[], Vector2d r_next[], Vector2d v_next[], int n_p){
//     double cos_phi = cos(100./180*M_PI);
//     double tau_a_inv = 1 / tau_a;
//     double sigma_inv = 1/ sigma;

//     for (int i = 0; i < n_p; i++) {
//             pedestrian *pi = &pedestrians[i];
//             Vector2d F_alpha(.0, .0);
            
//             // acceleration term
//             Vector2d F0_alpha = (pi->v0 * pi->e - pi->v) * tau_a_inv;

//             F_alpha += F0_alpha; 

//             // repulsive pedestrian
//             Vector2d F_ab(0., 0.);
//             for (int j = 0; j < n_p; j++) {
//                 Vector2d f_ab(.0, .0);
//                 pedestrian *pb = &pedestrians[j];
//                 if (pi == pb) {
//                     continue;
//                 }
//                 Vector2d rab = pi->r - pb->r;
//                 Vector2d vb = pb->v*Dt;
//                 Vector2d rabvb = rab - vb;
//                 double b = sqrt((rab.norm()+rabvb.norm())*(rab.norm()+rabvb.norm()) - vb.norm()*vb.norm()) * 0.5;

//                 f_ab = vab0*sigma_inv*exp(-b*sigma_inv) * (rab.norm()+rabvb.norm())*(rab/rab.norm()+rabvb/rabvb.norm()) / (4*b); 

//                 // weight
//                 double wef;
//                 if (pi->e.dot(-f_ab) >= -f_ab.norm()*cos_phi)
//                     wef = 1.;
//                 else
//                     wef = c;
//                 Vector2d F_ab_temp = wef * f_ab;
//                 F_ab += F_ab_temp;

//             }
//             F_alpha += F_ab; 

//             Vector2d raB0(0., pi->r.y);
//             Vector2d F_aB0 = UaB0/R*exp(-raB0.norm()/R) * raB0/raB0.norm(); 
//             Vector2d raB1(0., pi->r.y-height);
//             Vector2d F_aB1 = UaB0/R*exp(-raB1.norm()/R) * raB1/raB1.norm(); 

//             F_alpha += F_aB0 + F_aB1; 

//             r_next[i] += pi->v*dt;
//             Vector2d w_alpha = pi->v + F_alpha*dt;
//             v_next[i] = w_alpha * g(pi->vMax, w_alpha.norm());
//         }
//     return make_pair(r_next[n_p], v_next[n_p]);
// }

// now the unnecessary mem accesses are apparent, need optimization