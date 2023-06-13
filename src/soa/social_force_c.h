
#include <math.h>
#include <stdlib.h>

# define M_PI		3.14159265358979323846
// Map Size
const double width=8, height=4;

typedef struct Vector2d{
    double x, y;
} Vector2d;

Vector2d create_vector(double x, double y) {
    Vector2d new_vector;
    new_vector.x = x;
    new_vector.y = y;
    return new_vector;
}

// Vector operations
Vector2d vector_add(Vector2d v1, Vector2d v2) {
    Vector2d result = {v1.x + v2.x, v1.y + v2.y};
    return result;
}

Vector2d vector_sub(Vector2d v1, Vector2d v2) {
    Vector2d result = {v1.x - v2.x, v1.y - v2.y};
    return result;
}

Vector2d vector_div(Vector2d v, double a) {
    Vector2d result = {v.x / a, v.y / a};
    return result;
}

Vector2d vector_mult(Vector2d v, double a) {
    Vector2d result = {v.x * a, v.y * a};
    return result;
}

double vector_norm(Vector2d v) {
    return sqrt(v.x*v.x + v.y*v.y);
}

double vector_dot(Vector2d v1, Vector2d v2) {
    return v1.x*v2.x + v1.y*v2.y;
}

Vector2d inv (Vector2d v) {
    Vector2d result = {-v.x, -v.y};
    return result;
}

int n_p; 
int *color;

struct Vector2d *r, *r_next, *v, *v_next, *e;

double *v0, *vMax;

    
void init(int np) {
    n_p = np;
    color = (int *) malloc (sizeof(int) * n_p);
    r = (Vector2d *) malloc (sizeof(Vector2d) * n_p);
    r_next = (Vector2d *) malloc (sizeof(Vector2d) * n_p);
    v = (Vector2d *) malloc (sizeof(Vector2d) * n_p);
    v_next = (Vector2d *) malloc (sizeof(Vector2d) * n_p);
    e = (Vector2d *) malloc (sizeof(Vector2d) * n_p);
    v0 = (double *) malloc (sizeof(double) * n_p); 
    vMax = (double *) malloc (sizeof(double) * n_p);
    
    // initial velocity
    for (int i = 0; i < n_p; i++)
    {   
        v[i] = create_vector(0.0, 0.0);
        color[i] = rand() % 2;
        r[i] = create_vector((width * rand()) / RAND_MAX, (height * rand()) / RAND_MAX);

        r_next[i] = r[i];
        v_next[i] = v[i];
        e[i] = create_vector((pow(-1, color[i])), 0.0);
        v0[i] = 1 - 2 * rand() / RAND_MAX;
        vMax[i] = 1.3*v0[i];            
    }
    
}
    
// constants for simulation
const double Dt = 2, tau_a = 0.5, vab0 = 2.1, sigma = 0.3, phi = 100./180*M_PI, c = 0.5, dt = 0.01;
const double UaB0 = 10, R = 0.2;

// direction dependent weight
double w(Vector2d e, Vector2d f) {
    if (vector_dot(e, f) >= vector_norm(f)*cos(phi)) // 10
        return 1.;
    else
        return c;
}

double g(double v_max, double w_n) {
    if (w_n<=v_max)
        return 1.;
    else
        return v_max/w_n;
}

Vector2d Acceleration_term(int i) {
    return vector_div(vector_sub(e[i], v[i]),  tau_a * v0[i]);
}

double compute_b(double rabNorm, double rabvbNorm, double vbNorm) {
    return sqrt((rabNorm+rabvbNorm)*(rabNorm+rabvbNorm) - vbNorm*vbNorm)/2;
}
Vector2d compute_fab(double b, Vector2d rab, Vector2d rabvb) {
    return vector_mult(vector_div(vector_add(vector_div(rab, vector_norm(rab)), vector_div(rabvb, vector_norm(rabvb))),(4*b)), vab0/sigma*exp(-b/sigma) * (vector_norm(rab) + vector_norm(rabvb)));
    
}


Vector2d Repulsive_pedestrian(int i) { // 52n
    Vector2d F_rep_ped = create_vector(0., 0.);

    for (int j=0; j<n_p; j++) {
        if (i==j) {
            continue;
        }
        Vector2d rab = vector_sub(r[i], r[j]);
        Vector2d vb = vector_mult(v[j], Dt);
        Vector2d rabvb = vector_sub(rab, vb);

        // double b = sqrt((rab.norm()+rabvb.norm())*(rab.norm()+rabvb.norm()) - vb.norm()*vb.norm())/2; // 4x3+7=19
        double b = compute_b(vector_norm(rab) , vector_norm(rabvb), vector_norm(vb));

        // Vector2d f_ab = vab0/sigma*exp(-b/sigma) * (rab.norm()+rabvb.norm())*(rab/rab.norm()+rabvb/rabvb.norm()) / (4*b); // 12
        Vector2d f_ab = compute_fab(b, rab, rabvb);

        double wef = w(e[i], inv(f_ab)); // 11
        Vector2d F_ab = vector_mult(f_ab, wef);
        F_rep_ped = vector_add(F_ab, F_rep_ped);
        // cout<<rx[i]<<endl;

    }
    return F_rep_ped;
}

Vector2d Repulsive_border(Vector2d raB) {

    Vector2d FaB =  vector_mult(vector_div(raB, vector_norm(raB)), UaB0/R*exp(vector_norm(inv(raB))/R)); // 7 + 4x2 = 15
    return FaB;

}

// void update(ofstream& myfile){

//         // update location and velocity
//         for (int i = 0; i < n_p; i++)
//         {
//             // cout<<pi->r<<endl;
//             r[i] = r_next[i];
//             v[i] = v_next[i];
//             // pi->update();
//             if (r[i].x > width) {
//                 r[i].x -= width;
//             } else if (r[i].x < 0){
//                 r[i].x += width;
//             }
            
//             if (r[i].y > height) {
//                 r[i].y -= height;
//             } else if (r[i].y < 0){
//                 r[i].y += height;
//             }
//             myfile << r[i].x << " " << r[i].y <<" "<<color[i]<<endl;
//         }
// }


void _free(int* color, Vector2d* r, Vector2d* r_next, Vector2d* v, Vector2d* v_next, Vector2d* e, double* v0, double* vMax){
    free(color);
    free(r);
    free(r_next);
    free(v);
    free(v_next);
    free(e);
    free(v0);
    free(vMax);
}