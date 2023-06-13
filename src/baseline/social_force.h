#include <iostream>
#include <math.h>

using namespace std;
// Map Size
const double width=8, height=4;
struct Vector2d {
    double x, y;
    Vector2d(double a, double b) : x(a), y(b) {} 
    Vector2d() : x(.0), y(.0) {} 
    Vector2d operator+(Vector2d v) const {
        return Vector2d(v.x+x, v.y+y);
    }
    Vector2d operator-(Vector2d v) const {
        return Vector2d(x-v.x, y-v.y);
    }
    Vector2d operator/(double a) const {
        return Vector2d(x/a, y/a);
    }
    Vector2d operator*(double a) const {
        return Vector2d(x*a, y*a);
    }
    Vector2d operator+=(const Vector2d& v) {
        *this = *this + v;
        return *this;
    }
    double norm() {
        return sqrt(x*x+y*y);
    }
    double dot(Vector2d v) {
        return x*v.x+y*v.y;
    }
    
};


ostream& operator<<(ostream& os, const Vector2d& v) {
    os << v.x << " " << v.y;
    return os;
}
Vector2d operator- (Vector2d v) {
    return Vector2d(-v.x, -v.y);
}
Vector2d operator*(double a, Vector2d v) {
    return v*a;
}

class pedestrian {
public:
    int stateCurrent, stateFuture, color;
    Vector2d r, v, e;
    double v0, vMax;
    
    pedestrian() {
        // initial velocity
        v = Vector2d(0, 0);
        color = rand() % 2;
        r = Vector2d((width * rand()) / RAND_MAX, (height * rand()) / RAND_MAX);
        e = Vector2d(pow(-1, color), 0);
        v0 = 1 - 2 * rand() / RAND_MAX;
        vMax = 1.3*v0;
    }
    
    // in case pedestrian exceeds boundary
    void update() {
        if (r.x > width) {
            r.x -= width;
        } else if (r.x < 0){
            r.x += width;
        }
        
        if (r.y > height) {
            r.y -= height;
        } else if (r.y < 0){
            r.y += height;
        }
    }
};

// constants for simulation
const double Dt = 2, tau_a = 0.5, vab0 = 2.1, sigma = 0.3, phi = 100./180*M_PI, c = 0.5, dt = 0.01;
const double UaB0 = 10, R = 0.2;

// direction dependent weight
double w(Vector2d e, Vector2d f) {
    if (e.dot(f) >= f.norm()*cos(phi)) // 10
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

Vector2d Acceleration_term(pedestrian *p) {
    return (p->v0 * p->e - p->v) / tau_a;
}

Vector2d Repulsive_pedestrian(pedestrian *pa, pedestrian ps[], int n_p) { // 52n
    Vector2d F_rep_ped(0., 0.);

    for (int j=0; j<n_p; j++) {
        Vector2d f_ab(.0, .0);
        pedestrian *pb = &ps[j];
        if (pa==pb) {
            continue;
        }
        Vector2d rab = pa->r - pb->r;
        Vector2d vb = pb->v*Dt;
        Vector2d rabvb = rab - vb;

        double b = sqrt((rab.norm()+rabvb.norm())*(rab.norm()+rabvb.norm()) - vb.norm()*vb.norm())/2; // 4x3+7=19

        f_ab = vab0/sigma*exp(-b/sigma) * (rab.norm()+rabvb.norm())*(rab/rab.norm()+rabvb/rabvb.norm()) / (4*b); // 12

        double wef = w(pa->e, -f_ab); // 11
        Vector2d F_ab = wef * f_ab;
        F_rep_ped += F_ab;

    }
    return F_rep_ped;
}

Vector2d Repulsive_border(Vector2d raB) {
    Vector2d FaB = UaB0/R*exp(-raB.norm()/R) * raB/raB.norm(); // 7 + 4x2 = 15
    return FaB;

}

void update(pedestrian pedestrians[], Vector2d r_next[], Vector2d v_next[], ofstream& myfile, int n_p){

        // update location and velocity
        for (int i = 0; i < n_p; i++)
        {
            pedestrian *pi = &pedestrians[i];
            // cout<<pi->r<<endl;
            pi->r = r_next[i];
            pi->v = v_next[i];
            pi->update();
            myfile << pi->r.x << " " << pi->r.y<<" "<<pi->color<<endl;
        }
}