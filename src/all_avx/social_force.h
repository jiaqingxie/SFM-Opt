#ifndef SOCIAL_FORCE_H
#define SOCIAL_FORCE_H
#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#define UPDATE_FILE
#define PREFIX "./results/"
#define SUFFIX ".txt"

using namespace std;
const double width = 8, height = 4;
struct Vector2d {
    double x, y;
    Vector2d(double a, double b) : x(a), y(b) {}
    Vector2d() : x(.0), y(.0) {}
    Vector2d operator+(Vector2d v) const {
        return Vector2d(v.x + x, v.y + y);
    }
    Vector2d operator-(Vector2d v) const {
        return Vector2d(x - v.x, y - v.y);
    }
    Vector2d operator/(double a) const {
        return Vector2d(x / a, y / a);
    }
    Vector2d operator*(double a) const {
        return Vector2d(x * a, y * a);
    }
    Vector2d operator+=(const Vector2d& v) {
        *this = *this + v;
        return *this;
    }
    double norm() {
        return sqrt(x * x + y * y);
    }
    double dot(Vector2d v) {
        return x * v.x + y * v.y;
    }
};

ostream& operator<<(ostream& os, const Vector2d& v) {
    os << v.x << " " << v.y;
    return os;
}
Vector2d operator-(Vector2d v) {
    return Vector2d(-v.x, -v.y);
}
Vector2d operator*(double a, Vector2d v) {
    return v * a;
}

int n_p;
int* color;
Vector2d* r, * r_next, * v, * v_next, * e;
double* v0, * vMax;
double* r_next_x; double* r_next_y;
double* v_next_x; double* v_next_y;
double* v_x; double* v_y;
double* r_x; double* r_y;
double* e_x; double* e_y; 

void init(int np) {
    srand(666);
    n_p = np;
    color = new int[n_p];
    r = new Vector2d[n_p];
    r_next = new Vector2d[n_p];
    v = new Vector2d[n_p];
    v_next = new Vector2d[n_p];
    e = new Vector2d[n_p];
    r_x = new double[n_p];
    r_y = new double[n_p];
    v_x = new double[n_p];
    v_y = new double[n_p];
    r_next_x  = new double[n_p];
    r_next_y  = new double[n_p];
    v_next_x  = new double[n_p];
    v_next_y  = new double[n_p];
    e_x = new double[n_p];
    e_y = new double[n_p];

    v0 = new double[n_p];
    vMax = new double[n_p];
    // initial velocity
    for (int i = 0; i < n_p; i++) {
        // v[i] = Vector2d(1.0, 1.0);
        // color[i] = 2;
        // r[i] = Vector2d(0.0, 0.0);

        // r_next[i] = r[i];
        // v_next[i] = v[i];
        // e[i] = Vector2d(pow(-1, color[i]), 0.0);
        // v0[i] = 1 - 2 * 2 / 1;
        // vMax[i] = 1.3 * v0[i];
        v[i] = Vector2d(0.0, 0.0);
        v_x[i] = v[i].x;
        v_y[i] = v[i].y;
        color[i] = rand() % 2;
        r[i] = Vector2d((width * rand()) / RAND_MAX, (height * rand()) / RAND_MAX);

        r_next[i] = r[i];
        

        r_x[i] = r[i].x;
        r_y[i] = r[i].y;

        v_next[i] = v[i];
        v_next_x[i] = v[i].x;
        v_next_y[i] = v[i].y;

        e[i] = Vector2d(pow(-1, color[i]), 0.0);
        e_x[i] = e[i].x;
        e_y[i] = e[i].y;

        v0[i] = 1 - 2 * rand() / RAND_MAX;
        vMax[i] = 1.3 * v0[i];
    }
}

// constants for simulation
const double Dt = 2, tau_a = 0.5, vab0 = 2.1, sigma = 0.3, phi = 100. / 180 * M_PI, c = 0.5, dt = 0.01;
const double UaB0 = 10, R = 0.2;

// direction dependent weight
double w(Vector2d e, Vector2d f) {
    if (e.dot(f) >= f.norm() * cos(phi)) // 10
        return 1.;
    else
        return c;
}

double g(double v_max, double w_n) {
    if (w_n <= v_max)
        return 1.;
    else
        return v_max / w_n;
}

Vector2d Acceleration_term(int i) {
    return (e[i] - v[i]) / tau_a * v0[i];
}

double compute_b(double rabNorm, double rabvbNorm, double vbNorm) {
    return sqrt((rabNorm + rabvbNorm) * (rabNorm + rabvbNorm) - vbNorm * vbNorm) / 2;
}
Vector2d compute_fab(double b, Vector2d rab, Vector2d rabvb) {
    return vab0 / sigma * exp(-b / sigma) * (rab.norm() + rabvb.norm()) * (rab / rab.norm() + rabvb / rabvb.norm()) / (4 * b);
}

Vector2d Repulsive_pedestrian(int i) { // 52n
    Vector2d F_rep_ped(0., 0.);

    for (int j = 0; j < n_p; j++) {
        if (i == j) {
            continue;
        }
        Vector2d rab = r[i] - r[j];
        Vector2d vb = v[j] * Dt;
        Vector2d rabvb = rab - vb;

        // double b = sqrt((rab.norm()+rabvb.norm())*(rab.norm()+rabvb.norm()) - vb.norm()*vb.norm())/2; // 4x3+7=19
        double b = compute_b(rab.norm(), rabvb.norm(), vb.norm());

        // Vector2d f_ab = vab0/sigma*exp(-b/sigma) * (rab.norm()+rabvb.norm())*(rab/rab.norm()+rabvb/rabvb.norm()) / (4*b); // 12
        Vector2d f_ab = compute_fab(b, rab, rabvb);

        double wef = w(e[i], -f_ab); // 11
        Vector2d F_ab = wef * f_ab;
        F_rep_ped += F_ab;
        // cout<<rx[i]<<endl;
    }
    return F_rep_ped;
}

Vector2d Repulsive_border(Vector2d raB) {
    Vector2d FaB = UaB0 / R * exp(-raB.norm() / R) * raB / raB.norm(); // 7 + 4x2 = 15
    return FaB;
}

void update(ofstream& myfile) {
    // update location and velocity
    for (int i = 0; i < n_p; i++) {
        // cout<<pi->r<<endl;
        // v_x[i] = v_next_x[i];
        // v_y[i] = v_next_y[i];

        
        // pi->update();
        if (r_x[i] > width) {
            r_x[i] -= width;
        }
        else if (r_x[i] < 0) {
            r_x[i] += width;
        }

        if (r_y[i] > height) {
            r_y[i] -= height;
        }
        else if (r_y[i] < 0) {
            r_y[i] += height;
        }
#ifdef UPDATE_FILE
        myfile << r_x[i] << " " << r_y[i] << " " << color[i] << endl;
#endif
    }
}

double compute_error(string filename, int n_p) {
    string base = "baseline";
    string baseline_path = PREFIX + base + SUFFIX;
    ifstream baseline_file(baseline_path);

    string other_func_path = PREFIX + filename + SUFFIX;
    ifstream target_func_file(other_func_path);

    double error = 0.;

    if (baseline_file.fail()) {
        cout << "baseline file not found" << endl;
        exit(EXIT_FAILURE);
    }
    if (target_func_file.fail()) {
        cout << "target file not found" << endl;
        exit(EXIT_FAILURE);
    }

    double x1, y1, x2, y2;
    int c1, c2;
    for (int i = 0; i < n_p; i++) {
        baseline_file >> x1 >> y1 >> c1;
        target_func_file >> x2 >> y2 >> c2;
        // cout << x1 << " " << y1 << " " << c1 << endl;
        // cout << x2 << " " << y2 << " " << c2 << endl;

        // computing the euclidean distance as error
        error += sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    }

    error = isnan(error) ? INFINITY : error;

    // cout << "error of version " << filename << ": " << error << endl;

    return error;
}
#endif // SOCIAL_FORCE_H
