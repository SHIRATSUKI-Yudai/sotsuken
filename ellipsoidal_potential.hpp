#pragma once
#include <particle_simulator.hpp>
constexpr PS::F64 PI = MY_LIB::CONSTANT::pi;
/*
#include <iostream>
#include <random>

std::mt19937 eng(10);
std::uniform_real_distribution<double> rnd(-1.0, 1.0);

constexpr PS::F64 PI = 3.14159265358979323846;

struct Ptcl {
    double m;
    PS::F64vec pos;
    PS::F64vec vel;
};

template <typename Tptcl>
void gen_ptcl(Tptcl ptcl, const PS::S64 n_ptcl, const double a, const double b, const double c) {
    PS::S64 cnt = 0;
    while (cnt < n_ptcl) {
        double x = a * rnd(eng);
        double y = b * rnd(eng);
        double z = c * rnd(eng);
        if (x * x / (a * a) + y * y / (b * b) + z * z / (c * c) < 1.0) {
            ptcl[cnt].pos = PS::F64vec(x, y, z);
            ptcl[cnt].m = 1.0 / n_ptcl;
            cnt++;
        }
    }
}

void calc_force(PS::F64vec& acc, PS::F64& pot, const PS::F64vec pos, const Ptcl* ptcl,
                const PS::S64 n_ptcl) {
    acc = 0.0;
    pot = 0.0;
    for (PS::S64 i = 0; i < n_ptcl; i++) {
        PS::F64vec dr = pos - ptcl[i].pos;
        double r2 = dr * dr;
        double r_inv = 1.0 / sqrt(r2);
        pot -= ptcl[i].m * r_inv;
        acc -= ptcl[i].m * r_inv * r_inv * r_inv * dr;
    }
}
*/

void calc_mm(PS::F64vec& acc, PS::F64& pot, const PS::F64vec& pos, const PS::F64 mass, const PS::F64vec ellip) {
    //acc = 0.0;
    //pot = 0.0;
    const PS::F64 a = ellip.x;
    const PS::F64 b = ellip.y;
    const PS::F64 c = ellip.z;
    const PS::F64 x = pos.x;
    const PS::F64 y = pos.y;
    const PS::F64 z = pos.z;
    const PS::F64 x_sq = x * x;
    const PS::F64 y_sq = y * y;
    const PS::F64 z_sq = z * z;
    const PS::F64 rho = 3.0 * mass / (4.0 * PI * a * b * c);
    const PS::F64 r_sq = pos * pos;
    const PS::F64 r_inv = 1.0 / sqrt(r_sq);
    const PS::F64 pot0 = -mass * r_inv;
    const PS::F64 r2_inv = r_inv * r_inv;
    const PS::F64 r3_inv = r2_inv * r_inv;
    const PS::F64 r5_inv = r3_inv * r2_inv;
    const PS::F64vec acc0 = -mass * r3_inv * pos;
    const auto c_x = 2.0 * a * a - b * b - c * c;
    const auto c_y = 2.0 * b * b - c * c - a * a;
    const auto c_z = 2.0 * c * c - a * a - b * b;
    const PS::F64 qxx = 4.0 * PI * rho / 15.0 * a * b * c * c_x;
    const PS::F64 qyy = 4.0 * PI * rho / 15.0 * a * b * c * c_y;
    const PS::F64 qzz = 4.0 * PI * rho / 15.0 * a * b * c * c_z;
    const PS::F64 qrr = qxx * x_sq + qyy * y_sq + qzz * z_sq;
    const PS::F64 pot2 = -0.5 * r5_inv * qrr;
    pot += pot0 + pot2;
    const PS::F64vec tmp_v(qxx * x, qyy * y, qzz * z);
    const PS::F64vec acc2 = 0.5 * r5_inv * r2_inv * (-5.0 * qrr * pos + 2.0 * r_sq * tmp_v);
    acc = acc0 + acc2;
}

void calc_force_rot(PS::F64vec& acc, PS::F64& pot, const PS::F64vec& pos, const PS::F64vec& vel,
                    const PS::F64 mass, const PS::F64vec ellip, const PS::F64vec& omega) {
    //acc = 0.0;
    //pot = 0.0;
    PS::F64vec acc_pot = 0.0;
    calc_mm(acc_pot, pot, pos, mass, ellip);
    //PS::F64vec vel_0 = vel - (omega ^ pos);
    PS::F64vec acc_cor = -2.0 * (omega ^ vel);
    PS::F64vec acc_cen = -omega ^ (omega ^ pos);
    acc += acc_pot + acc_cor + acc_cen;
    // std::cout<<std::endl;
    // std::cout << "omega ^ pos= " << (omega ^ pos) << std::endl;
    // std::cout << "acc_pot= " << acc_pot << std::endl;
    // std::cout << "acc_cor= " << acc_cor << std::endl;
    // std::cout << "acc_cen= " << acc_cen << std::endl;
    // std::cout << "acc= " << acc << std::endl;
}
/*
PS::F64vec calc_vel_kep(const PS::F64 mass, const PS::F64vec& pos) {
    const PS::F64 r_sq = pos * pos;
    const PS::F64 r_inv = 1.0 / sqrt(r_sq);
    const PS::F64vec er = pos * r_inv;
    const PS::F64vec ez(0.0, 0.0, 1.0);
    const PS::F64vec ev = ez ^ er;
    return sqrt(mass * r_inv) * ev;
}

void update(PS::F64vec& pos, PS::F64vec& vel, PS::F64vec& acc, PS::F64& pot, const PS::F64 dt,
            const PS::F64 rho, const PS::F64 a, const PS::F64 b, const PS::F64 c,
            const PS::F64vec& omega) {
    vel += acc * 0.5 * dt;
    pos += vel * dt;
    calc_force_rot(acc, pot, pos, vel, rho, a, b, c, omega);
    vel += acc * 0.5 * dt;
}

int main() {
    std::cout << std::setprecision(15);
    // constexpr double a = 1.0;
    // constexpr double b = 0.5;
    // constexpr double c = 0.3;
    //  https://en.wikipedia.org/wiki/Quaoar
    //constexpr auto a = 1.0;
    //constexpr auto b = a / 1.19;
    //constexpr auto c = b / 1.16;
    constexpr auto a = 1.0;
    constexpr auto b = 1.0;
    constexpr auto c = 1.0;
    constexpr auto r = 1.0;
    constexpr auto mass = 1.0;
    const PS::F64vec omega(0.0, 0.0, 4.0*sqrt(mass / (r * r * r)));
    constexpr auto rho = mass / (4.0 * PI / 3.0 * a * b * c);
    constexpr PS::S64 n_ptcl = 1 << 20;
    Ptcl* ptcl = new Ptcl[n_ptcl];
    gen_ptcl(ptcl, n_ptcl, a, b, c);
    PS::F64vec pos_rot(r, 0.0, 0.0);
    PS::F64vec acc_rot(0.0, 0.0, 0.0);
    PS::F64vec vel = calc_vel_kep(mass, pos_rot);
    PS::F64vec vel_rot = vel - (omega ^ pos_rot);
    PS::F64 pot = 0.0;
    calc_force_rot(acc_rot, pot, pos_rot, vel_rot, rho, a, b, c, omega);
    const double dt = 1.0 / 131072.0;
    const double t_kep = 2.0 * PI * sqrt(r * r * r / mass);
    //const double t_end = t_kep * 10.0;
    const double t_end = t_kep;
    double time = 0.0;
    const PS::S32 n_loop_max = t_end / dt;
    for (PS::S32 i = 0; i < n_loop_max; i++) {
        if (i % 100 == 0) {
            std::cout << "time= " << time << " pos_rot= " << pos_rot << " vel_rot= " << vel_rot
                      << " acc_rot= " << acc_rot << " pot= " << pot << std::endl;
        }
        update(pos_rot, vel_rot, acc_rot, pot, dt, rho, a, b, c, omega);
        time += dt;
    }

#if 0    
    calc_force(acc, pot, pos, ptcl, n_ptcl);
    PS::F64vec acc_exp = 0.0;
    PS::F64 pot_exp = 0.0;
    calc_mm(acc_exp, pot_exp, pos, rho, a, b, c);
    std::cout << "pos= " << pos << " acc= " << acc << " pot= " << pot << std::endl;
    std::cout << "pos= " << pos << " acc_exp= " << acc_exp << " pot_exp= " << pot_exp << std::endl;
    pos = PS::F64vec(0.0, 10.0, 0.0);
    calc_force(acc, pot, pos, ptcl, n_ptcl);
    calc_mm(acc_exp, pot_exp, pos, rho, a, b, c);
    std::cout << "pos= " << pos << " acc= " << acc << " pot= " << pot << std::endl;
    std::cout << "pos= " << pos << " acc_exp= " << acc_exp << " pot_exp= " << pot_exp << std::endl;
    pos = PS::F64vec(0.0, 0.0, 10.0);
    calc_force(acc, pot, pos, ptcl, n_ptcl);
    calc_mm(acc_exp, pot_exp, pos, rho, a, b, c);
    std::cout << "pos= " << pos << " acc= " << acc << " pot= " << pot << std::endl;
    std::cout << "pos= " << pos << " acc_exp= " << acc_exp << " pot_exp= " << pot_exp << std::endl;
#endif
    return 0;
}
*/