#include <iostream>
#include <cstdio>
#include <algorithm>
#include <unistd.h>
#include <random>
#include <particle_simulator.hpp>
#include "user_defined.hpp"
#include "./fdps-util/fdps-util.hpp"
#include "./fdps-util/my_lib.hpp"
#include "./fdps-util/kepler.hpp"
#include "./fdps-util/init.hpp"

constexpr PS::F64 MY_PI = MY_LIB::CONSTANT::pi;
FP_t PLANET;

class DiskInfo
{
    PS::F64 ax_;
    PS::F64 delta_ax_;
    PS::F64 r_hill_;
    PS::F64 r_phy_;
    PS::F64 dens_;
    PS::F64 m_ptcl_;
    PS::F64 kappa_;
    PS::F64 eta_;
    PS::F64 e_refl_;
    PS::F64 t_dur_;
    PS::F64 tau_;
    PS::S64 n_glb_;
    PS::F64 ax_in_;
    PS::F64 ax_out_;
    void setParticlesOnLayer(std::vector<PS::F64vec> &pos, const PS::S64 id_x_head, const PS::S64 id_x_tail, const PS::S64 id_y_head, const PS::S64 id_y_tail, const PS::F64 dtheta, const PS::F64 dr, const PS::F64ort box,
                             const PS::F64 offset_theta = 0.0, const PS::F64 offset_r = 0.0, const PS::F64 offset_z = 0.0, const PS::F64 eps = 0.0)
    {
        static std::mt19937 mt(PS::Comm::getRank());
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
        PS::F64 pos_z = offset_z;
        for (auto i = id_x_head - 3; i <= id_x_tail + 3; i++)
        {
            PS::F64 pos_x = ((PS::F64)i) * dtheta + offset_theta;
            if (pos_x < box.low_.x || pos_x >= box.high_.x)
                continue;
            for (auto j = id_y_head - 3; j <= id_y_tail + 3; j++)
            {
                PS::F64 pos_y = j * dr + offset_r;
                if (pos_y < box.low_.y || pos_y >= box.high_.y)
                    continue;
                PS::F64 eps_x = eps * (dist(mt) - 0.5) * 2.0;
                PS::F64 eps_y = eps * (dist(mt) - 0.5) * 2.0;
                PS::F64 eps_z = eps * (dist(mt) - 0.5) * 2.0;
                pos.push_back(PS::F64vec(pos_x + eps_x, pos_y + eps_y, pos_z + eps_z));
            }
        }
    }

    void calcKappa()
    {
        kappa_ = std::pow((2.0 * MY_PI / t_dur_), 2.0); // k^{'};
    }
    void calcEta()
    {
        const auto ln_e_refl = std::log(e_refl_);
        eta_ = 4.0 * MY_PI / (t_dur_ * std::sqrt(1.0 + std::pow((MY_PI / ln_e_refl), 2.0))); // \eta^{'}
                                                                                             // return 4.0*MY_PI*ln_e_refl/(t_dur_*std::sqrt(MY_PI*MY_PI+ln_e_refl)); // \eta^{'}
    }

public:
    void setParams(const PS::F64 delta_ax, const PS::F64 e_refl, const PS::F64 t_dur, const PS::F64 tau, const PS::F64 rphy_over_rhill, const PS::S64 n_glb, const PS::F64 ax_in, const PS::F64 ax_out, const PS::F64 dens,const PS::F64 m_ptcl)
    {

        ax_ = (ax_in + ax_out) / 2.0;
        delta_ax_ = delta_ax;
        e_refl_ = e_refl;
        t_dur_ = t_dur;
        tau_ = tau;
        n_glb_ = n_glb;
        ax_in_ = ax_in;
        ax_out_ = ax_out;
        //r_phy_ = sqrt(tau_ * (ax_out_ * ax_out_ - ax_in_ * ax_in_) / n_glb_);
        //r_phy_ = CM2REARTH(pow(m_ptcl * 5.97e27 / (4.0 / 3.0 * MY_PI * 3.3), 1.0 / 3.0));
	    r_phy_ = pow(m_ptcl, 1.0 / 3.0 ) / 2.456 * 2.9;
        r_hill_ = r_phy_ / rphy_over_rhill;
        m_ptcl_ = m_ptcl;
        // m_ptcl_ = (r_hill_ / ((ax_out_ + ax_in_) * 0.5)) * (r_hill_ / ((ax_out_ + ax_in_) * 0.5)) * (r_hill_ / ((ax_out_ + ax_in_) * 0.5)) * 3.0 * PLANET.mass * 0.5;
        dens_ = dens;
        // dens_ = m_ptcl_ * n_glb / (MY_PI * (ax_out_ * ax_out_ - ax_in_ * ax_in_));
        calcKappa();
        calcEta();
    }
    DiskInfo() {}
    DiskInfo(const PS::F64 delta_ax, const PS::F64 e_refl, const PS::F64 t_dur, const PS::F64 tau, const PS::F64 rphy_over_rhill, const PS::S64 n_glb, const PS::F64 ax_in, const PS::F64 ax_out, const PS::F64 dens,const PS::F64 mass_planet_glb)
    {
        setParams(delta_ax, e_refl, t_dur, tau, rphy_over_rhill, n_glb, ax_in, ax_out,dens,mass_planet_glb);
    }
    void writeAscii(FILE *fp)
    {
        fprintf(fp, "%12.11e   %12.11e   %12.11e   %12.11e   %12.11e   %12.11e   %12.11e   %12.11e   %12.11e   %12.11e   %12.11e %lld \n",
                ax_, delta_ax_, r_hill_, r_phy_, dens_, m_ptcl_, kappa_, eta_, e_refl_, t_dur_, tau_, n_glb_);
    }
    void readAscii(FILE *fp)
    {
        auto tmp = fscanf(fp, "%lf   %lf   %lf   %lf   %lf   %lf   %lf   %lf   %lf   %lf   %lf %lld",
                          &ax_, &delta_ax_, &r_hill_, &r_phy_, &dens_, &m_ptcl_, &kappa_, &eta_, &e_refl_, &t_dur_, &tau_, &n_glb_);
        assert(tmp == 12);
        FP_t::kappa = kappa_;
        FP_t::eta = eta_;
    }

    void writeBinary(FILE *fp)
    {
        fwrite(this, sizeof(*this), 1, fp);
    }
    void readBinary(FILE *fp)
    {
        auto tmp = fread(this, sizeof(*this), 1, fp);
        assert(tmp == 1);
        FP_t::kappa = kappa_;
        FP_t::eta = eta_;
    }

public:
    template <class Tpsys>
    void set_r_coll_search(Tpsys &psys,PS::S64 n){
        for (PS::S64 i = 0; i < n; i++)
        {
            psys[i].r_coll = r_phy_;
            psys[i].r_search = 6.0 * r_hill_;
        }
    }

    template <class Tpsys>
    PS::S64 setParticles(Tpsys &psys, const PS::F64ort box, const bool layer = true, const bool random_shift = true)
    {
        //const auto ax_in = ax_ - 0.5 * delta_ax_;
        //const auto ax_out = ax_ + 0.5 * delta_ax_;
        const auto area = MY_PI * (ax_out_ * ax_out_ - ax_in_ * ax_in_);
        PS::F64 dS = area / n_glb_;
        if (layer)
        {
            dS *= 3.0;
        }
        const auto dl = sqrt(dS);
        assert(delta_ax_ > dl);
        const auto dz = dl;
        const PS::S64 n_theta = (2.0 * MY_PI * ax_ / dl);
        const auto dtheta = 2.0 * MY_PI / n_theta;
        const PS::S64 n_r = (delta_ax_ / dl);
        const auto dr = delta_ax_ / n_r;

        const PS::S64 id_x_head = box.low_.x / dtheta;
        const PS::S64 id_x_tail = box.high_.x / dtheta;
        const PS::S64 id_y_head = box.low_.y / dr;
        const PS::S64 id_y_tail = box.high_.y / dr;

        PS::F64 eps = 0.0;
        if (random_shift)
        {
            eps = (dl > 2.0 * r_phy_) ? (0.5 * (dl - 2.0 * r_phy_)) * 0.9 : 0.0;
        }

        PS::Comm::barrier();
        if (PS::Comm::getRank() == 0)
        {
            std::cerr << "delta_ax_= " << delta_ax_
                      << " dS = " << dS
                      << " dl = " << dl
                      << " n_r= " << n_r
                      << " dr= " << dr
                      << " eps= " << eps
                      << std::endl;
            std::cerr << "n_theta= " << n_theta
                      << " dtheta= " << dtheta
                      << std::endl;
        }
        PS::Comm::barrier();
        std::vector<PS::F64vec> pos;
        PS::F64 offset_theta = dtheta * (sqrt(2.0) - 1.0) * 0.5;
        PS::F64 offset_r = 0.0;
        PS::F64 offset_z = 0.0;
        setParticlesOnLayer(pos, id_x_head, id_x_tail, id_y_head, id_y_tail, dtheta, dr, box, offset_theta, offset_r, offset_z, eps);
        if (layer == true)
        {
            offset_theta = 0.5 * dtheta + dtheta * (sqrt(2.0) - 1.0) * 0.5;
            offset_r = 0.5 * dr;
            offset_z = -0.5 * dz;
            setParticlesOnLayer(pos, id_x_head, id_x_tail, id_y_head, id_y_tail, dtheta, dr, box, offset_theta, offset_r, offset_z, eps);
            offset_z = 0.5 * dz;
            setParticlesOnLayer(pos, id_x_head, id_x_tail, id_y_head, id_y_tail, dtheta, dr, box, offset_theta, offset_r, offset_z, eps);
        }

        PS::S64 n_loc = pos.size();
        psys.setNumberOfParticleLocal(n_loc);
        for (PS::S64 i = 0; i < n_loc; i++)
        {
            psys[i].pos_cyl = pos[i];
            psys[i].pos_car = ConvertCyl2Car(pos[i]);
            psys[i].vel = GetVel(psys[i].pos_car);
            psys[i].mass = m_ptcl_;
            psys[i].r_coll = r_phy_;
            psys[i].r_search = 6.0 * r_hill_;
            psys[i].eps = 0.0;
            psys[i].kappa = kappa_;
            psys[i].eta = eta_;
        }
        return n_loc;
    }
};

template<typename Tpsys>
void CalcForceFromPlanet(Tpsys & psys, const FP_t & pla){
    const auto n = psys.getNumberOfParticleLocal();
#pragma omp parallel for
    for(auto i=0; i<n; i++){
        const auto rij = psys[i].pos_car - pla.pos_car;
        const auto r_sq = rij*rij;
        const auto r_inv = 1.0 / sqrt(r_sq);
        const auto pot   = pla.mass * r_inv;
        psys[i].acc -= pot * r_inv * r_inv * rij;
        psys[i].pot -= pot;
    }
}

int main(int argc, char *argv[])
{
    PS::Initialize(argc, argv);

    PS::ParticleSystem<FP_t> system;
    system.initialize();
    PS::S64 n_glb = 1.0e5;  // 粒子数
    PS::F64 ax_in = 1;    // 地球とリングの内側までの距離[REARTH]
    PS::F64 ax_out = 3.5;     // 地球とリングの外側までの距離[REARTH]
    PS::F64 ecc_rms = 0.3;  // normalized
    PS::F64 inc_rms = 0.15; // normalized
    PS::F64 dens = 1.0e8;   // [g/cm^2]
    PS::F64 mass_sun = 1.0; //[MEARTH]
    double a_ice = 0.0;
    double f_ice = 1.0;
    double power = -3.0;
    PS::S32 seed = 0;

    PS::S64 n_loc = n_glb;

    PS::F64 m_ptcl;

    PLANET.mass = mass_sun;
    PLANET.pos_car = PLANET.vel = 0.0;

    SetParticleKeplerDisk(system, n_glb, ax_in, ax_out, ecc_rms, inc_rms, dens, mass_sun, a_ice, f_ice, power, seed);
    m_ptcl = system[0].mass;
    if (PS::Comm::getRank() == 0)
    {

        FILE *fp;
        fp = fopen("./result/temp/snap00000.dat", "w");

        DiskInfo disk_info;
        FDPS_UTIL::ComandLineOptionManager cmd(argc, argv);
        cmd.append("delta_ax", "a", "the width of semi^major axis", "1.0e-3");
        cmd.append("inv_dt", "d", "inverse of dt", "256");
        cmd.append("theta", "t", "iopening criterion", "0.5");
        cmd.append("time_end", "T", "ending time", "10.0");
        cmd.append("n_leaf_limit", "l", "n_leaf_limit", "8");
        cmd.append("n_group_limit", "n", "maximum # of particles in a leaf cell", "64");
        cmd.append("n_step_share", "L", "# of steps sharing the same interaction list", "1");
        cmd.append("n_loc", "N", "# of particles per process", "32768");
        cmd.append("ex_let_mode", "e", "exchange LET mode \n  0: PS::EXCHANGE_LET_A2A \n  1: PS::EXCHANGE_LET_P2P_EXACT \n  2: PS::EXCHANGE_LET_P2P_FAST \n", "0");
        cmd.append("e_refl", "E", "restitution coefficient", "0.5");
        cmd.append("t_dur", "D", "inverse of duration time", "4");
        cmd.append("tau", "O", "optical depth", "1.0");
        cmd.append("rphy_over_rhill", "R", "rphy / rhill", "1.0");
        cmd.appendFlag("flag_mtm", "m", "use mid point tree method");
        cmd.appendFlag("flag_rot", "r", "use rotation method");
        cmd.appendFlag("flag_para_out", "output files in parallel");
        cmd.appendFlag("flag_bin_out", "output binary format");
        cmd.appendNoDefault("read_file", "i", "write file name base", false);
        cmd.appendNoDefault("write_file", "o", "write file name base", false);
        cmd.append("sat_mode", "satellite mode\n  0: no satellite\n  1: PAN\n  2: few satellites\n", "0");
        cmd.append("sat_mass_ratio", "satellite-mass / particle-mass (this option is available ONLY IF sat_mode=2)", "8.0");
        cmd.append("sat_num", "satellite number per process (this option is available ONLY IF sat_mode=2)", "10");
        cmd.append("n_smp", "# of sample particles per process", "100");
        cmd.append("dt_snp", "the interval time of snapshot", "1.0");
        cmd.appendNoDefault("log_file", "logfile name", false);
        cmd.read();

        PS::F64 delta_ax = ax_out - ax_in;
        PS::F64 e_refl = 0.1;
        PS::F64 t_dur = cmd.get("t_dur");
        PS::F64 tau = cmd.get("tau");
        PS::F64 rphy_over_rhill = cmd.get("rphy_over_rhill");
        disk_info.setParams(delta_ax, e_refl, t_dur, tau, rphy_over_rhill, n_glb, ax_in, ax_out,dens,m_ptcl);
    
        CalcForceFromPlanet(system, PLANET);

        Energy eng_now;
        eng_now.calc(system);

        SatelliteSystem sat_system_glb;
        sat_system_glb.setId();

        FDPS_UTIL::SnapshotManager::HeaderParam header_param(0, system.getNumberOfParticleLocal(), system.getNumberOfParticleGlobal(), 0);
        FDPS_UTIL::FileHeader<FDPS_UTIL::SnapshotManager::HeaderParam, DiskInfo, Energy, SatelliteSystem> file_header(header_param, disk_info, eng_now, sat_system_glb);
        file_header.writeAscii(fp);

        disk_info.set_r_coll_search(system, n_glb);

        for (int i = 0; i < n_glb; i++){
            system[i].writeAscii(fp);
        }

        fclose(fp);

        PS::F64 dens_p = m_ptcl * 5.97e27 / ((4.0 / 3.0) * MY_PI * pow(REARTH2CM(system[0].r_coll), 3));
        PS::F64 dens_e = 5.97e27 / ((4.0 / 3.0) * MY_PI * pow(REARTH2CM(1.0), 3.0));

        std::cout << "R_Roche=" << 2.456 * pow(dens_p / dens_e, -1.0 / 3.0) << std::endl;
        std::cout << "particles density = " << dens_p << std::endl;
        std::cout << "earth density = " << dens_e << std::endl;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /*
                fp = fopen("./result/test/check.dat", "w");

                std::vector<double> r(n_glb);
                for (int i = 0; i < n_glb;i++){
                    // system[i].pos_cyl.x
                    // r[i] = sqrt(system[i].getPosCar().x*system[i].getPosCar().x+
                    //             system[i].getPosCar().y*system[i].getPosCar().y+
                    //             system[i].getPosCar().z*system[i].getPosCar().z);
                    r[i] = ConvertCar2Cyl(system[i].pos_car).y;
                }

                sort(r.begin(), r.end());
                int i = 1;
                for (auto a : r){
                    fprintf(fp, "%d %f\n", i, a);
                    i++;
                }
                fclose(fp);
        */
    }
    return 0;
}
