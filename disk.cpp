#include<iostream>
#include<cstdio>
#include<algorithm>
#include<unistd.h>
#include<random>
#include<particle_simulator.hpp>
#include"user_defined.hpp"
#include"./fdps-util/my_lib.hpp"
#include"./fdps-util/kepler.hpp"
#include"./fdps-util/init.hpp"

int main(int argc, char *argv[]){
    PS::Initialize(argc, argv);

    PS::ParticleSystem<FP_t> system;
    system.initialize();
    PS::S64 n_glb=1.0e5;    //粒子数
    PS::F64 ax_in=0.5;              // 太陽とリングの内側までの距離[AU]
    PS::F64 ax_out=1.5;             // 太陽とリングの外側までの距離[AU]
    PS::F64 ecc_rms = 0.0;          // normalized
    PS::F64 inc_rms = 0.0;         // normalized
    // PS::F64 dens = 10.0;        // [g/cm^2]
    // PS::F64 mass_sun = 1.0;     //[m_sun]
    // double a_ice = 0.0;
    // double f_ice = 1.0;
    // double power = -1.5;
    // PS::S32 seed = 0;

    PS::S64 n_loc = n_glb;

    SetParticleKeplerDisk(system,n_glb,ax_in,ax_out,ecc_rms,inc_rms);

    if(PS::Comm::getRank()==0){

        FILE *fp;
        fp = fopen("./result/test/snap00000.dat", "w");

        //fprintf(fp, "0.00000000000e+00  %lld  %lld  0\n", n_loc, n_glb);
        for (int i = 0; i < n_glb; i++)
            system[i].writeAscii(fp);
        fclose(fp);

        //FILE *fp;
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
    }
    return 0;
}