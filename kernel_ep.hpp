#include<pikg_vector.hpp>
#include<cmath>
#include<limits>
#include<chrono>

#include <pikg_avx512.hpp>
struct CalcForceEpEpImpl{
PIKG::F32 eps2;
PIKG::F32 kappa;
PIKG::F32 eta;
CalcForceEpEpImpl(){}
CalcForceEpEpImpl(PIKG::F32 eps2,PIKG::F32 kappa,PIKG::F32 eta):eps2(eps2),kappa(kappa),eta(eta){}
void initialize(PIKG::F32 eps2_,PIKG::F32 kappa_,PIKG::F32 eta_){
eps2 = eps2_;
kappa = kappa_;
eta = eta_;
}
int kernel_id = 0;
void operator()(const Epi0* __restrict__ epi,const int ni,const Epj0* __restrict__ epj,const int nj,Force0* __restrict__ force,const int kernel_select = 1){
static_assert(sizeof(Epi0) == 36,"check consistency of EPI member variable definition between PIKG source and original source");
static_assert(sizeof(Epj0) == 36,"check consistency of EPJ member variable definition between PIKG source and original source");
static_assert(sizeof(Force0) == 28,"check consistency of FORCE member variable definition between PIKG source and original source");
if(kernel_select>=0) kernel_id = kernel_select;
if(kernel_id == 0){
std::cout << "ni: " << ni << " nj:" << nj << std::endl;
Force0* force_tmp = new Force0[ni];
std::chrono::system_clock::time_point  start, end;
double min_time = std::numeric_limits<double>::max();
{ // test Kernel_I16_J1
for(int i=0;i<ni;i++) force_tmp[i] = force[i];
start = std::chrono::system_clock::now();
Kernel_I16_J1(epi,ni,epj,nj,force_tmp);
end = std::chrono::system_clock::now();
double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
std::cerr << "kerel 1: " << elapsed << " ns" << std::endl;
if(min_time > elapsed){
min_time = elapsed;
kernel_id = 1;
}
}
{ // test Kernel_I1_J16
for(int i=0;i<ni;i++) force_tmp[i] = force[i];
start = std::chrono::system_clock::now();
Kernel_I1_J16(epi,ni,epj,nj,force_tmp);
end = std::chrono::system_clock::now();
double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
std::cerr << "kerel 2: " << elapsed << " ns" << std::endl;
if(min_time > elapsed){
min_time = elapsed;
kernel_id = 2;
}
}
delete[] force_tmp;
} // if(kernel_id == 0)
if(kernel_id == 1) Kernel_I16_J1(epi,ni,epj,nj,force);
if(kernel_id == 2) Kernel_I1_J16(epi,ni,epj,nj,force);
} // operator() definition 
void Kernel_I16_J1(const Epi0* __restrict__ epi,const PIKG::S32 ni,const Epj0* __restrict__ epj,const PIKG::S32 nj,Force0* __restrict__ force){
PIKG::S32 i;
PIKG::S32 j;
for(i = 0;i < (ni/16)*16;i += 16){
__m512 EPI_mass;

alignas(32) int32_t index_gather_load0[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load0 = _mm512_load_epi32(index_gather_load0);
EPI_mass = _mm512_i32gather_ps(vindex_gather_load0,((float*)&epi[i+0].mass),4);
__m512x3 EPI_pos;

alignas(32) int32_t index_gather_load1[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load1 = _mm512_load_epi32(index_gather_load1);
EPI_pos.v0 = _mm512_i32gather_ps(vindex_gather_load1,((float*)&epi[i+0].pos.x),4);
alignas(32) int32_t index_gather_load2[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load2 = _mm512_load_epi32(index_gather_load2);
EPI_pos.v1 = _mm512_i32gather_ps(vindex_gather_load2,((float*)&epi[i+0].pos.y),4);
alignas(32) int32_t index_gather_load3[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load3 = _mm512_load_epi32(index_gather_load3);
EPI_pos.v2 = _mm512_i32gather_ps(vindex_gather_load3,((float*)&epi[i+0].pos.z),4);
__m512 EPI_r_coll;

alignas(32) int32_t index_gather_load4[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load4 = _mm512_load_epi32(index_gather_load4);
EPI_r_coll = _mm512_i32gather_ps(vindex_gather_load4,((float*)&epi[i+0].r_coll),4);
__m512x3 EPI_vel;

alignas(32) int32_t index_gather_load5[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load5 = _mm512_load_epi32(index_gather_load5);
EPI_vel.v0 = _mm512_i32gather_ps(vindex_gather_load5,((float*)&epi[i+0].vel.x),4);
alignas(32) int32_t index_gather_load6[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load6 = _mm512_load_epi32(index_gather_load6);
EPI_vel.v1 = _mm512_i32gather_ps(vindex_gather_load6,((float*)&epi[i+0].vel.y),4);
alignas(32) int32_t index_gather_load7[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load7 = _mm512_load_epi32(index_gather_load7);
EPI_vel.v2 = _mm512_i32gather_ps(vindex_gather_load7,((float*)&epi[i+0].vel.z),4);
__m512x3 FORCE_acc;

FORCE_acc.v0 = _mm512_set1_ps(0.0f);
FORCE_acc.v1 = _mm512_set1_ps(0.0f);
FORCE_acc.v2 = _mm512_set1_ps(0.0f);
__m512x3 FORCE_acc_dash;

FORCE_acc_dash.v0 = _mm512_set1_ps(0.0f);
FORCE_acc_dash.v1 = _mm512_set1_ps(0.0f);
FORCE_acc_dash.v2 = _mm512_set1_ps(0.0f);
__m512 FORCE_pot;

FORCE_pot = _mm512_set1_ps(0.0f);
for(j = 0;j < (nj/1)*1;++j){
__m512 EPJ_mass;

EPJ_mass = _mm512_set1_ps(epj[j].mass);

__m512x3 EPJ_pos;

EPJ_pos.v0 = _mm512_set1_ps(epj[j].pos.x);

EPJ_pos.v1 = _mm512_set1_ps(epj[j].pos.y);

EPJ_pos.v2 = _mm512_set1_ps(epj[j].pos.z);

__m512 EPJ_r_coll;

EPJ_r_coll = _mm512_set1_ps(epj[j].r_coll);

__m512x3 EPJ_vel;

EPJ_vel.v0 = _mm512_set1_ps(epj[j].vel.x);

EPJ_vel.v1 = _mm512_set1_ps(epj[j].vel.y);

EPJ_vel.v2 = _mm512_set1_ps(epj[j].vel.z);

__m512x3 acc_sprg_tmp;

__m512x3 acc_dash_tmp;

__m512x3 acc_grav_tmp;

__m512 pot_sprg_tmp;

__m512 pot_grav_tmp;

__m512x3 rij;

__m512 __fkg_tmp10;

__m512 __fkg_tmp9;

__m512 r_real_sq;

__m512 r_coll_tmp;

__m512 r_coll_sq;

__m512 over_r_real;

__m512 over_r_real_sq;

__m512 r_coll_cu;

__m512 over_r_coll_cu;

__m512 tmp0;

__m512 __fkg_tmp0;

__m512x3 __fkg_tmp4;

__m512 pot_offset;

__m512 __fkg_tmp5;

__m512 m_red;

__m512 r_real;

__m512 dr;

__m512 __fkg_tmp1;

__m512x3 __fkg_tmp6;

__m512 __fkg_tmp7;

__m512x3 vij;

__m512 rv;

__m512 __fkg_tmp2;

__m512x3 __fkg_tmp8;

__m512 m_over_r_real;

__m512 tmp1;

__m512 __fkg_tmp3;

__m512 __fkg_tmp12;

__m512 __fkg_tmp11;

__m512 __fkg_tmp14;

__m512 __fkg_tmp13;

__m512 __fkg_tmp16;

__m512 __fkg_tmp15;

__m512 __fkg_tmp17;

acc_sprg_tmp.v0 = _mm512_set1_ps(0.0f);
acc_sprg_tmp.v1 = _mm512_set1_ps(0.0f);
acc_sprg_tmp.v2 = _mm512_set1_ps(0.0f);
acc_dash_tmp.v0 = _mm512_set1_ps(0.0f);
acc_dash_tmp.v1 = _mm512_set1_ps(0.0f);
acc_dash_tmp.v2 = _mm512_set1_ps(0.0f);
acc_grav_tmp.v0 = _mm512_set1_ps(0.0f);
acc_grav_tmp.v1 = _mm512_set1_ps(0.0f);
acc_grav_tmp.v2 = _mm512_set1_ps(0.0f);
pot_sprg_tmp = _mm512_set1_ps(0.0f);
pot_grav_tmp = _mm512_set1_ps(0.0f);
rij.v0 = _mm512_sub_ps(EPI_pos.v0,EPJ_pos.v0);
rij.v1 = _mm512_sub_ps(EPI_pos.v1,EPJ_pos.v1);
rij.v2 = _mm512_sub_ps(EPI_pos.v2,EPJ_pos.v2);
__fkg_tmp10 = _mm512_fmadd_ps(rij.v0,rij.v0,_mm512_set1_ps(eps2));
__fkg_tmp9 = _mm512_fmadd_ps(rij.v1,rij.v1,__fkg_tmp10);
r_real_sq = _mm512_fmadd_ps(rij.v2,rij.v2,__fkg_tmp9);
{
__mmask16 pg1;
__mmask16 pg0;
pg1 = _mm512_cmp_ps_mask(r_real_sq,_mm512_set1_ps(eps2),_CMP_NEQ_OQ);
pg0 = pg1;

r_coll_tmp = _mm512_add_ps(EPI_r_coll,EPJ_r_coll);
r_coll_sq = _mm512_mul_ps(r_coll_tmp,r_coll_tmp);
over_r_real = rsqrt(r_real_sq);
over_r_real_sq = _mm512_mul_ps(over_r_real,over_r_real);
{
__mmask16 pg3;
__mmask16 pg2;
pg3 = _mm512_cmp_ps_mask(r_real_sq,r_coll_sq,_CMP_LT_OQ);
pg2 = pg3;
pg3 = _kand_mask16(pg3,pg1);

r_coll_cu = _mm512_mul_ps(r_coll_sq,r_coll_tmp);
over_r_coll_cu = _mm512_div_ps(_mm512_set1_ps(1.0f),r_coll_cu);
tmp0 = _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(1.0f)),EPJ_mass);
__fkg_tmp0 = _mm512_mul_ps(tmp0,over_r_coll_cu);
__fkg_tmp4.v0 = _mm512_mul_ps(_mm512_mul_ps(tmp0,over_r_coll_cu),rij.v0);
__fkg_tmp4.v1 = _mm512_mul_ps(_mm512_mul_ps(tmp0,over_r_coll_cu),rij.v1);
__fkg_tmp4.v2 = _mm512_mul_ps(_mm512_mul_ps(tmp0,over_r_coll_cu),rij.v2);
pot_offset = _mm512_div_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(1.5f)),r_coll_tmp);
__fkg_tmp5 = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.25f),EPJ_mass),_mm512_fmadd_ps(r_real_sq,over_r_coll_cu,pot_offset));
m_red = _mm512_div_ps(EPJ_mass,_mm512_add_ps(EPI_mass,EPJ_mass));
r_real = _mm512_mul_ps(r_real_sq,over_r_real);
dr = _mm512_sub_ps(r_coll_tmp,r_real);
__fkg_tmp1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real);
__fkg_tmp6.v0 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real),rij.v0);
__fkg_tmp6.v1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real),rij.v1);
__fkg_tmp6.v2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real),rij.v2);
__fkg_tmp7 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.25f),_mm512_set1_ps(kappa)),m_red),dr),dr);
vij.v0 = _mm512_sub_ps(EPI_vel.v0,EPJ_vel.v0);
vij.v1 = _mm512_sub_ps(EPI_vel.v1,EPJ_vel.v1);
vij.v2 = _mm512_sub_ps(EPI_vel.v2,EPJ_vel.v2);
rv = _mm512_fmadd_ps(rij.v2,vij.v2,_mm512_fmadd_ps(rij.v0,vij.v0,_mm512_mul_ps(rij.v1,vij.v1)));
__fkg_tmp2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq);
__fkg_tmp8.v0 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq),rij.v0);
__fkg_tmp8.v1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq),rij.v1);
__fkg_tmp8.v2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq),rij.v2);
acc_grav_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v0,__fkg_tmp4.v0);;
acc_grav_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v1,__fkg_tmp4.v1);;
acc_grav_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v2,__fkg_tmp4.v2);;
pot_grav_tmp = _mm512_mask_blend_ps(pg3,pot_grav_tmp,__fkg_tmp5);;
acc_sprg_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_sprg_tmp.v0,__fkg_tmp6.v0);;
acc_sprg_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_sprg_tmp.v1,__fkg_tmp6.v1);;
acc_sprg_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_sprg_tmp.v2,__fkg_tmp6.v2);;
pot_sprg_tmp = _mm512_mask_blend_ps(pg3,pot_sprg_tmp,__fkg_tmp7);;
acc_dash_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_dash_tmp.v0,__fkg_tmp8.v0);;
acc_dash_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_dash_tmp.v1,__fkg_tmp8.v1);;
acc_dash_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_dash_tmp.v2,__fkg_tmp8.v2);;
pg3 = _knot_mask16(pg2);
pg3 = _kand_mask16(pg3,pg1);

m_over_r_real = _mm512_mul_ps(EPJ_mass,over_r_real);
tmp1 = _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(1.0f)),m_over_r_real);
__fkg_tmp3 = _mm512_mul_ps(tmp1,over_r_real_sq);
__fkg_tmp4.v0 = _mm512_mul_ps(_mm512_mul_ps(tmp1,over_r_real_sq),rij.v0);
__fkg_tmp4.v1 = _mm512_mul_ps(_mm512_mul_ps(tmp1,over_r_real_sq),rij.v1);
__fkg_tmp4.v2 = _mm512_mul_ps(_mm512_mul_ps(tmp1,over_r_real_sq),rij.v2);
__fkg_tmp5 = _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(0.5f)),m_over_r_real);
acc_grav_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v0,__fkg_tmp4.v0);;
acc_grav_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v1,__fkg_tmp4.v1);;
acc_grav_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v2,__fkg_tmp4.v2);;
pot_grav_tmp = _mm512_mask_blend_ps(pg3,pot_grav_tmp,__fkg_tmp5);;
}

}

__fkg_tmp12 = _mm512_add_ps(acc_grav_tmp.v0,acc_sprg_tmp.v0);
__fkg_tmp11 = _mm512_add_ps(__fkg_tmp12,acc_dash_tmp.v0);
FORCE_acc.v0 = _mm512_add_ps(FORCE_acc.v0,__fkg_tmp11);
__fkg_tmp14 = _mm512_add_ps(acc_grav_tmp.v1,acc_sprg_tmp.v1);
__fkg_tmp13 = _mm512_add_ps(__fkg_tmp14,acc_dash_tmp.v1);
FORCE_acc.v1 = _mm512_add_ps(FORCE_acc.v1,__fkg_tmp13);
__fkg_tmp16 = _mm512_add_ps(acc_grav_tmp.v2,acc_sprg_tmp.v2);
__fkg_tmp15 = _mm512_add_ps(__fkg_tmp16,acc_dash_tmp.v2);
FORCE_acc.v2 = _mm512_add_ps(FORCE_acc.v2,__fkg_tmp15);
FORCE_acc_dash.v0 = _mm512_add_ps(FORCE_acc_dash.v0,acc_dash_tmp.v0);
FORCE_acc_dash.v1 = _mm512_add_ps(FORCE_acc_dash.v1,acc_dash_tmp.v1);
FORCE_acc_dash.v2 = _mm512_add_ps(FORCE_acc_dash.v2,acc_dash_tmp.v2);
__fkg_tmp17 = _mm512_add_ps(pot_grav_tmp,pot_sprg_tmp);
FORCE_pot = _mm512_add_ps(FORCE_pot,__fkg_tmp17);
} // loop of j

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load8[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_gather_load8 = _mm512_load_epi32(index_gather_load8);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load8,((float*)&force[i+0].acc.x),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_acc.v0);
int32_t index_scatter_store0[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_scatter_store0 = _mm512_load_epi32(index_scatter_store0);
_mm512_i32scatter_ps(((float*)&force[i+0].acc.x),vindex_scatter_store0,__fkg_tmp_accum,4);
}

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load9[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_gather_load9 = _mm512_load_epi32(index_gather_load9);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load9,((float*)&force[i+0].acc.y),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_acc.v1);
int32_t index_scatter_store1[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_scatter_store1 = _mm512_load_epi32(index_scatter_store1);
_mm512_i32scatter_ps(((float*)&force[i+0].acc.y),vindex_scatter_store1,__fkg_tmp_accum,4);
}

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load10[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_gather_load10 = _mm512_load_epi32(index_gather_load10);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load10,((float*)&force[i+0].acc.z),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_acc.v2);
int32_t index_scatter_store2[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_scatter_store2 = _mm512_load_epi32(index_scatter_store2);
_mm512_i32scatter_ps(((float*)&force[i+0].acc.z),vindex_scatter_store2,__fkg_tmp_accum,4);
}

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load11[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_gather_load11 = _mm512_load_epi32(index_gather_load11);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load11,((float*)&force[i+0].acc_dash.x),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_acc_dash.v0);
int32_t index_scatter_store3[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_scatter_store3 = _mm512_load_epi32(index_scatter_store3);
_mm512_i32scatter_ps(((float*)&force[i+0].acc_dash.x),vindex_scatter_store3,__fkg_tmp_accum,4);
}

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load12[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_gather_load12 = _mm512_load_epi32(index_gather_load12);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load12,((float*)&force[i+0].acc_dash.y),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_acc_dash.v1);
int32_t index_scatter_store4[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_scatter_store4 = _mm512_load_epi32(index_scatter_store4);
_mm512_i32scatter_ps(((float*)&force[i+0].acc_dash.y),vindex_scatter_store4,__fkg_tmp_accum,4);
}

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load13[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_gather_load13 = _mm512_load_epi32(index_gather_load13);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load13,((float*)&force[i+0].acc_dash.z),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_acc_dash.v2);
int32_t index_scatter_store5[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_scatter_store5 = _mm512_load_epi32(index_scatter_store5);
_mm512_i32scatter_ps(((float*)&force[i+0].acc_dash.z),vindex_scatter_store5,__fkg_tmp_accum,4);
}

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load14[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_gather_load14 = _mm512_load_epi32(index_gather_load14);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load14,((float*)&force[i+0].pot),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_pot);
int32_t index_scatter_store6[16] = {0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105};
__m512i vindex_scatter_store6 = _mm512_load_epi32(index_scatter_store6);
_mm512_i32scatter_ps(((float*)&force[i+0].pot),vindex_scatter_store6,__fkg_tmp_accum,4);
}

} // loop of i
{ // tail loop of reference 
for(;i < ni;++i){
PIKG::F32 EPI_mass;

EPI_mass = epi[i+0].mass;
PIKG::F32vec EPI_pos;

EPI_pos.x = epi[i+0].pos.x;
EPI_pos.y = epi[i+0].pos.y;
EPI_pos.z = epi[i+0].pos.z;
PIKG::F32 EPI_r_coll;

EPI_r_coll = epi[i+0].r_coll;
PIKG::F32vec EPI_vel;

EPI_vel.x = epi[i+0].vel.x;
EPI_vel.y = epi[i+0].vel.y;
EPI_vel.z = epi[i+0].vel.z;
PIKG::F32vec FORCE_acc;

FORCE_acc.x = 0.0f;
FORCE_acc.y = 0.0f;
FORCE_acc.z = 0.0f;
PIKG::F32vec FORCE_acc_dash;

FORCE_acc_dash.x = 0.0f;
FORCE_acc_dash.y = 0.0f;
FORCE_acc_dash.z = 0.0f;
PIKG::F32 FORCE_pot;

FORCE_pot = 0.0f;
for(j = 0;j < nj;++j){
PIKG::F32 EPJ_mass;

EPJ_mass = epj[j].mass;
PIKG::F32vec EPJ_pos;

EPJ_pos.x = epj[j].pos.x;
EPJ_pos.y = epj[j].pos.y;
EPJ_pos.z = epj[j].pos.z;
PIKG::F32 EPJ_r_coll;

EPJ_r_coll = epj[j].r_coll;
PIKG::F32vec EPJ_vel;

EPJ_vel.x = epj[j].vel.x;
EPJ_vel.y = epj[j].vel.y;
EPJ_vel.z = epj[j].vel.z;
PIKG::F32vec acc_sprg_tmp;

PIKG::F32vec acc_dash_tmp;

PIKG::F32vec acc_grav_tmp;

PIKG::F32 pot_sprg_tmp;

PIKG::F32 pot_grav_tmp;

PIKG::F32vec rij;

PIKG::F32 __fkg_tmp10;

PIKG::F32 __fkg_tmp9;

PIKG::F32 r_real_sq;

PIKG::F32 r_coll_tmp;

PIKG::F32 r_coll_sq;

PIKG::F32 over_r_real;

PIKG::F32 over_r_real_sq;

PIKG::F32 r_coll_cu;

PIKG::F32 over_r_coll_cu;

PIKG::F32 tmp0;

PIKG::F32 __fkg_tmp0;

PIKG::F32vec __fkg_tmp4;

PIKG::F32 pot_offset;

PIKG::F32 __fkg_tmp5;

PIKG::F32 m_red;

PIKG::F32 r_real;

PIKG::F32 dr;

PIKG::F32 __fkg_tmp1;

PIKG::F32vec __fkg_tmp6;

PIKG::F32 __fkg_tmp7;

PIKG::F32vec vij;

PIKG::F32 rv;

PIKG::F32 __fkg_tmp2;

PIKG::F32vec __fkg_tmp8;

PIKG::F32 m_over_r_real;

PIKG::F32 tmp1;

PIKG::F32 __fkg_tmp3;

PIKG::F32 __fkg_tmp12;

PIKG::F32 __fkg_tmp11;

PIKG::F32 __fkg_tmp14;

PIKG::F32 __fkg_tmp13;

PIKG::F32 __fkg_tmp16;

PIKG::F32 __fkg_tmp15;

PIKG::F32 __fkg_tmp17;

acc_sprg_tmp.x = 0.0f;
acc_sprg_tmp.y = 0.0f;
acc_sprg_tmp.z = 0.0f;
acc_dash_tmp.x = 0.0f;
acc_dash_tmp.y = 0.0f;
acc_dash_tmp.z = 0.0f;
acc_grav_tmp.x = 0.0f;
acc_grav_tmp.y = 0.0f;
acc_grav_tmp.z = 0.0f;
pot_sprg_tmp = 0.0f;
pot_grav_tmp = 0.0f;
rij.x = (EPI_pos.x-EPJ_pos.x);
rij.y = (EPI_pos.y-EPJ_pos.y);
rij.z = (EPI_pos.z-EPJ_pos.z);
__fkg_tmp10 = (rij.x*rij.x+eps2);
__fkg_tmp9 = (rij.y*rij.y+__fkg_tmp10);
r_real_sq = (rij.z*rij.z+__fkg_tmp9);
if((r_real_sq!=eps2)){
r_coll_tmp = (EPI_r_coll+EPJ_r_coll);
r_coll_sq = (r_coll_tmp*r_coll_tmp);
over_r_real = rsqrt(r_real_sq);
over_r_real_sq = (over_r_real*over_r_real);
if((r_coll_sq>r_real_sq)){
r_coll_cu = (r_coll_sq*r_coll_tmp);
over_r_coll_cu = (1.0f/r_coll_cu);
tmp0 = (-(1.0f)*EPJ_mass);
__fkg_tmp0 = (tmp0*over_r_coll_cu);
__fkg_tmp4.x = ((tmp0*over_r_coll_cu)*rij.x);
__fkg_tmp4.y = ((tmp0*over_r_coll_cu)*rij.y);
__fkg_tmp4.z = ((tmp0*over_r_coll_cu)*rij.z);
pot_offset = (-(1.5f)/r_coll_tmp);
__fkg_tmp5 = ((0.25f*EPJ_mass)*(r_real_sq*over_r_coll_cu+pot_offset));
m_red = (EPJ_mass/(EPI_mass+EPJ_mass));
r_real = (r_real_sq*over_r_real);
dr = (r_coll_tmp-r_real);
__fkg_tmp1 = (((kappa*m_red)*dr)*over_r_real);
__fkg_tmp6.x = ((((kappa*m_red)*dr)*over_r_real)*rij.x);
__fkg_tmp6.y = ((((kappa*m_red)*dr)*over_r_real)*rij.y);
__fkg_tmp6.z = ((((kappa*m_red)*dr)*over_r_real)*rij.z);
__fkg_tmp7 = ((((0.25f*kappa)*m_red)*dr)*dr);
vij.x = (EPI_vel.x-EPJ_vel.x);
vij.y = (EPI_vel.y-EPJ_vel.y);
vij.z = (EPI_vel.z-EPJ_vel.z);
rv = (rij.z*vij.z+(rij.x*vij.x+(rij.y*vij.y)));
__fkg_tmp2 = (((eta*m_red)*rv)*over_r_real_sq);
__fkg_tmp8.x = ((((eta*m_red)*rv)*over_r_real_sq)*rij.x);
__fkg_tmp8.y = ((((eta*m_red)*rv)*over_r_real_sq)*rij.y);
__fkg_tmp8.z = ((((eta*m_red)*rv)*over_r_real_sq)*rij.z);
acc_grav_tmp.x = __fkg_tmp4.x;
acc_grav_tmp.y = __fkg_tmp4.y;
acc_grav_tmp.z = __fkg_tmp4.z;
pot_grav_tmp = __fkg_tmp5;
acc_sprg_tmp.x = __fkg_tmp6.x;
acc_sprg_tmp.y = __fkg_tmp6.y;
acc_sprg_tmp.z = __fkg_tmp6.z;
pot_sprg_tmp = __fkg_tmp7;
acc_dash_tmp.x = __fkg_tmp8.x;
acc_dash_tmp.y = __fkg_tmp8.y;
acc_dash_tmp.z = __fkg_tmp8.z;
}else{
m_over_r_real = (EPJ_mass*over_r_real);
tmp1 = (-(1.0f)*m_over_r_real);
__fkg_tmp3 = (tmp1*over_r_real_sq);
__fkg_tmp4.x = ((tmp1*over_r_real_sq)*rij.x);
__fkg_tmp4.y = ((tmp1*over_r_real_sq)*rij.y);
__fkg_tmp4.z = ((tmp1*over_r_real_sq)*rij.z);
__fkg_tmp5 = (-(0.5f)*m_over_r_real);
acc_grav_tmp.x = __fkg_tmp4.x;
acc_grav_tmp.y = __fkg_tmp4.y;
acc_grav_tmp.z = __fkg_tmp4.z;
pot_grav_tmp = __fkg_tmp5;
}
}
__fkg_tmp12 = (acc_grav_tmp.x+acc_sprg_tmp.x);
__fkg_tmp11 = (__fkg_tmp12+acc_dash_tmp.x);
FORCE_acc.x = (FORCE_acc.x+__fkg_tmp11);
__fkg_tmp14 = (acc_grav_tmp.y+acc_sprg_tmp.y);
__fkg_tmp13 = (__fkg_tmp14+acc_dash_tmp.y);
FORCE_acc.y = (FORCE_acc.y+__fkg_tmp13);
__fkg_tmp16 = (acc_grav_tmp.z+acc_sprg_tmp.z);
__fkg_tmp15 = (__fkg_tmp16+acc_dash_tmp.z);
FORCE_acc.z = (FORCE_acc.z+__fkg_tmp15);
FORCE_acc_dash.x = (FORCE_acc_dash.x+acc_dash_tmp.x);
FORCE_acc_dash.y = (FORCE_acc_dash.y+acc_dash_tmp.y);
FORCE_acc_dash.z = (FORCE_acc_dash.z+acc_dash_tmp.z);
__fkg_tmp17 = (pot_grav_tmp+pot_sprg_tmp);
FORCE_pot = (FORCE_pot+__fkg_tmp17);
} // loop of j

force[i+0].acc.x = (force[i+0].acc.x+FORCE_acc.x);
force[i+0].acc.y = (force[i+0].acc.y+FORCE_acc.y);
force[i+0].acc.z = (force[i+0].acc.z+FORCE_acc.z);
force[i+0].acc_dash.x = (force[i+0].acc_dash.x+FORCE_acc_dash.x);
force[i+0].acc_dash.y = (force[i+0].acc_dash.y+FORCE_acc_dash.y);
force[i+0].acc_dash.z = (force[i+0].acc_dash.z+FORCE_acc_dash.z);
force[i+0].pot = (force[i+0].pot+FORCE_pot);
} // loop of i
} // end loop of reference 
} // Kernel_I16_J1 definition 
void Kernel_I1_J16(const Epi0* __restrict__ epi,const PIKG::S32 ni,const Epj0* __restrict__ epj,const PIKG::S32 nj,Force0* __restrict__ force){
PIKG::S32 i;
PIKG::S32 j;
for(i = 0;i < (ni/1)*1;++i){
__m512 EPI_mass;

EPI_mass = _mm512_set1_ps(epi[i+0].mass);

__m512x3 EPI_pos;

EPI_pos.v0 = _mm512_set1_ps(epi[i+0].pos.x);

EPI_pos.v1 = _mm512_set1_ps(epi[i+0].pos.y);

EPI_pos.v2 = _mm512_set1_ps(epi[i+0].pos.z);

__m512 EPI_r_coll;

EPI_r_coll = _mm512_set1_ps(epi[i+0].r_coll);

__m512x3 EPI_vel;

EPI_vel.v0 = _mm512_set1_ps(epi[i+0].vel.x);

EPI_vel.v1 = _mm512_set1_ps(epi[i+0].vel.y);

EPI_vel.v2 = _mm512_set1_ps(epi[i+0].vel.z);

__m512x3 FORCE_acc;

FORCE_acc.v0 = _mm512_set1_ps(0.0f);
FORCE_acc.v1 = _mm512_set1_ps(0.0f);
FORCE_acc.v2 = _mm512_set1_ps(0.0f);
__m512x3 FORCE_acc_dash;

FORCE_acc_dash.v0 = _mm512_set1_ps(0.0f);
FORCE_acc_dash.v1 = _mm512_set1_ps(0.0f);
FORCE_acc_dash.v2 = _mm512_set1_ps(0.0f);
__m512 FORCE_pot;

FORCE_pot = _mm512_set1_ps(0.0f);
for(j = 0;j < (nj/16)*16;j += 16){
__m512 EPJ_mass;

alignas(32) int32_t index_gather_load15[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load15 = _mm512_load_epi32(index_gather_load15);
EPJ_mass = _mm512_i32gather_ps(vindex_gather_load15,((float*)&epj[j].mass),4);
__m512x3 EPJ_pos;

alignas(32) int32_t index_gather_load16[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load16 = _mm512_load_epi32(index_gather_load16);
EPJ_pos.v0 = _mm512_i32gather_ps(vindex_gather_load16,((float*)&epj[j].pos.x),4);
alignas(32) int32_t index_gather_load17[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load17 = _mm512_load_epi32(index_gather_load17);
EPJ_pos.v1 = _mm512_i32gather_ps(vindex_gather_load17,((float*)&epj[j].pos.y),4);
alignas(32) int32_t index_gather_load18[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load18 = _mm512_load_epi32(index_gather_load18);
EPJ_pos.v2 = _mm512_i32gather_ps(vindex_gather_load18,((float*)&epj[j].pos.z),4);
__m512 EPJ_r_coll;

alignas(32) int32_t index_gather_load19[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load19 = _mm512_load_epi32(index_gather_load19);
EPJ_r_coll = _mm512_i32gather_ps(vindex_gather_load19,((float*)&epj[j].r_coll),4);
__m512x3 EPJ_vel;

alignas(32) int32_t index_gather_load20[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load20 = _mm512_load_epi32(index_gather_load20);
EPJ_vel.v0 = _mm512_i32gather_ps(vindex_gather_load20,((float*)&epj[j].vel.x),4);
alignas(32) int32_t index_gather_load21[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load21 = _mm512_load_epi32(index_gather_load21);
EPJ_vel.v1 = _mm512_i32gather_ps(vindex_gather_load21,((float*)&epj[j].vel.y),4);
alignas(32) int32_t index_gather_load22[16] = {0,9,18,27,36,45,54,63,72,81,90,99,108,117,126,135};
__m512i vindex_gather_load22 = _mm512_load_epi32(index_gather_load22);
EPJ_vel.v2 = _mm512_i32gather_ps(vindex_gather_load22,((float*)&epj[j].vel.z),4);
__m512x3 acc_sprg_tmp;

__m512x3 acc_dash_tmp;

__m512x3 acc_grav_tmp;

__m512 pot_sprg_tmp;

__m512 pot_grav_tmp;

__m512x3 rij;

__m512 __fkg_tmp10;

__m512 __fkg_tmp9;

__m512 r_real_sq;

__m512 r_coll_tmp;

__m512 r_coll_sq;

__m512 over_r_real;

__m512 over_r_real_sq;

__m512 r_coll_cu;

__m512 over_r_coll_cu;

__m512 tmp0;

__m512 __fkg_tmp0;

__m512x3 __fkg_tmp4;

__m512 pot_offset;

__m512 __fkg_tmp5;

__m512 m_red;

__m512 r_real;

__m512 dr;

__m512 __fkg_tmp1;

__m512x3 __fkg_tmp6;

__m512 __fkg_tmp7;

__m512x3 vij;

__m512 rv;

__m512 __fkg_tmp2;

__m512x3 __fkg_tmp8;

__m512 m_over_r_real;

__m512 tmp1;

__m512 __fkg_tmp3;

__m512 __fkg_tmp12;

__m512 __fkg_tmp11;

__m512 __fkg_tmp14;

__m512 __fkg_tmp13;

__m512 __fkg_tmp16;

__m512 __fkg_tmp15;

__m512 __fkg_tmp17;

acc_sprg_tmp.v0 = _mm512_set1_ps(0.0f);
acc_sprg_tmp.v1 = _mm512_set1_ps(0.0f);
acc_sprg_tmp.v2 = _mm512_set1_ps(0.0f);
acc_dash_tmp.v0 = _mm512_set1_ps(0.0f);
acc_dash_tmp.v1 = _mm512_set1_ps(0.0f);
acc_dash_tmp.v2 = _mm512_set1_ps(0.0f);
acc_grav_tmp.v0 = _mm512_set1_ps(0.0f);
acc_grav_tmp.v1 = _mm512_set1_ps(0.0f);
acc_grav_tmp.v2 = _mm512_set1_ps(0.0f);
pot_sprg_tmp = _mm512_set1_ps(0.0f);
pot_grav_tmp = _mm512_set1_ps(0.0f);
rij.v0 = _mm512_sub_ps(EPI_pos.v0,EPJ_pos.v0);
rij.v1 = _mm512_sub_ps(EPI_pos.v1,EPJ_pos.v1);
rij.v2 = _mm512_sub_ps(EPI_pos.v2,EPJ_pos.v2);
__fkg_tmp10 = _mm512_fmadd_ps(rij.v0,rij.v0,_mm512_set1_ps(eps2));
__fkg_tmp9 = _mm512_fmadd_ps(rij.v1,rij.v1,__fkg_tmp10);
r_real_sq = _mm512_fmadd_ps(rij.v2,rij.v2,__fkg_tmp9);
{
__mmask16 pg1;
__mmask16 pg0;
pg1 = _mm512_cmp_ps_mask(r_real_sq,_mm512_set1_ps(eps2),_CMP_NEQ_OQ);
pg0 = pg1;

r_coll_tmp = _mm512_add_ps(EPI_r_coll,EPJ_r_coll);
r_coll_sq = _mm512_mul_ps(r_coll_tmp,r_coll_tmp);
over_r_real = rsqrt(r_real_sq);
over_r_real_sq = _mm512_mul_ps(over_r_real,over_r_real);
{
__mmask16 pg3;
__mmask16 pg2;
pg3 = _mm512_cmp_ps_mask(r_real_sq,r_coll_sq,_CMP_LT_OQ);
pg2 = pg3;
pg3 = _kand_mask16(pg3,pg1);

r_coll_cu = _mm512_mul_ps(r_coll_sq,r_coll_tmp);
over_r_coll_cu = _mm512_div_ps(_mm512_set1_ps(1.0f),r_coll_cu);
tmp0 = _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(1.0f)),EPJ_mass);
__fkg_tmp0 = _mm512_mul_ps(tmp0,over_r_coll_cu);
__fkg_tmp4.v0 = _mm512_mul_ps(_mm512_mul_ps(tmp0,over_r_coll_cu),rij.v0);
__fkg_tmp4.v1 = _mm512_mul_ps(_mm512_mul_ps(tmp0,over_r_coll_cu),rij.v1);
__fkg_tmp4.v2 = _mm512_mul_ps(_mm512_mul_ps(tmp0,over_r_coll_cu),rij.v2);
pot_offset = _mm512_div_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(1.5f)),r_coll_tmp);
__fkg_tmp5 = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.25f),EPJ_mass),_mm512_fmadd_ps(r_real_sq,over_r_coll_cu,pot_offset));
m_red = _mm512_div_ps(EPJ_mass,_mm512_add_ps(EPI_mass,EPJ_mass));
r_real = _mm512_mul_ps(r_real_sq,over_r_real);
dr = _mm512_sub_ps(r_coll_tmp,r_real);
__fkg_tmp1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real);
__fkg_tmp6.v0 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real),rij.v0);
__fkg_tmp6.v1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real),rij.v1);
__fkg_tmp6.v2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real),rij.v2);
__fkg_tmp7 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.25f),_mm512_set1_ps(kappa)),m_red),dr),dr);
vij.v0 = _mm512_sub_ps(EPI_vel.v0,EPJ_vel.v0);
vij.v1 = _mm512_sub_ps(EPI_vel.v1,EPJ_vel.v1);
vij.v2 = _mm512_sub_ps(EPI_vel.v2,EPJ_vel.v2);
rv = _mm512_fmadd_ps(rij.v2,vij.v2,_mm512_fmadd_ps(rij.v0,vij.v0,_mm512_mul_ps(rij.v1,vij.v1)));
__fkg_tmp2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq);
__fkg_tmp8.v0 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq),rij.v0);
__fkg_tmp8.v1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq),rij.v1);
__fkg_tmp8.v2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq),rij.v2);
acc_grav_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v0,__fkg_tmp4.v0);;
acc_grav_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v1,__fkg_tmp4.v1);;
acc_grav_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v2,__fkg_tmp4.v2);;
pot_grav_tmp = _mm512_mask_blend_ps(pg3,pot_grav_tmp,__fkg_tmp5);;
acc_sprg_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_sprg_tmp.v0,__fkg_tmp6.v0);;
acc_sprg_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_sprg_tmp.v1,__fkg_tmp6.v1);;
acc_sprg_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_sprg_tmp.v2,__fkg_tmp6.v2);;
pot_sprg_tmp = _mm512_mask_blend_ps(pg3,pot_sprg_tmp,__fkg_tmp7);;
acc_dash_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_dash_tmp.v0,__fkg_tmp8.v0);;
acc_dash_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_dash_tmp.v1,__fkg_tmp8.v1);;
acc_dash_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_dash_tmp.v2,__fkg_tmp8.v2);;
pg3 = _knot_mask16(pg2);
pg3 = _kand_mask16(pg3,pg1);

m_over_r_real = _mm512_mul_ps(EPJ_mass,over_r_real);
tmp1 = _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(1.0f)),m_over_r_real);
__fkg_tmp3 = _mm512_mul_ps(tmp1,over_r_real_sq);
__fkg_tmp4.v0 = _mm512_mul_ps(_mm512_mul_ps(tmp1,over_r_real_sq),rij.v0);
__fkg_tmp4.v1 = _mm512_mul_ps(_mm512_mul_ps(tmp1,over_r_real_sq),rij.v1);
__fkg_tmp4.v2 = _mm512_mul_ps(_mm512_mul_ps(tmp1,over_r_real_sq),rij.v2);
__fkg_tmp5 = _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(0.5f)),m_over_r_real);
acc_grav_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v0,__fkg_tmp4.v0);;
acc_grav_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v1,__fkg_tmp4.v1);;
acc_grav_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v2,__fkg_tmp4.v2);;
pot_grav_tmp = _mm512_mask_blend_ps(pg3,pot_grav_tmp,__fkg_tmp5);;
}

}

__fkg_tmp12 = _mm512_add_ps(acc_grav_tmp.v0,acc_sprg_tmp.v0);
__fkg_tmp11 = _mm512_add_ps(__fkg_tmp12,acc_dash_tmp.v0);
FORCE_acc.v0 = _mm512_add_ps(FORCE_acc.v0,__fkg_tmp11);
__fkg_tmp14 = _mm512_add_ps(acc_grav_tmp.v1,acc_sprg_tmp.v1);
__fkg_tmp13 = _mm512_add_ps(__fkg_tmp14,acc_dash_tmp.v1);
FORCE_acc.v1 = _mm512_add_ps(FORCE_acc.v1,__fkg_tmp13);
__fkg_tmp16 = _mm512_add_ps(acc_grav_tmp.v2,acc_sprg_tmp.v2);
__fkg_tmp15 = _mm512_add_ps(__fkg_tmp16,acc_dash_tmp.v2);
FORCE_acc.v2 = _mm512_add_ps(FORCE_acc.v2,__fkg_tmp15);
FORCE_acc_dash.v0 = _mm512_add_ps(FORCE_acc_dash.v0,acc_dash_tmp.v0);
FORCE_acc_dash.v1 = _mm512_add_ps(FORCE_acc_dash.v1,acc_dash_tmp.v1);
FORCE_acc_dash.v2 = _mm512_add_ps(FORCE_acc_dash.v2,acc_dash_tmp.v2);
__fkg_tmp17 = _mm512_add_ps(pot_grav_tmp,pot_sprg_tmp);
FORCE_pot = _mm512_add_ps(FORCE_pot,__fkg_tmp17);
} // loop of j

if(j<nj){ // tail j loop
__m512x3 __fkg_tmp18;

__fkg_tmp18.v0 = FORCE_acc.v0;
__fkg_tmp18.v1 = FORCE_acc.v1;
__fkg_tmp18.v2 = FORCE_acc.v2;
__m512x3 __fkg_tmp19;

__fkg_tmp19.v0 = FORCE_acc_dash.v0;
__fkg_tmp19.v1 = FORCE_acc_dash.v1;
__fkg_tmp19.v2 = FORCE_acc_dash.v2;
__m512 __fkg_tmp20;

__fkg_tmp20 = FORCE_pot;
for(;j < nj;++j){
__m512 EPJ_mass;

EPJ_mass = _mm512_set1_ps(epj[j].mass);

__m512x3 EPJ_pos;

EPJ_pos.v0 = _mm512_set1_ps(epj[j].pos.x);

EPJ_pos.v1 = _mm512_set1_ps(epj[j].pos.y);

EPJ_pos.v2 = _mm512_set1_ps(epj[j].pos.z);

__m512 EPJ_r_coll;

EPJ_r_coll = _mm512_set1_ps(epj[j].r_coll);

__m512x3 EPJ_vel;

EPJ_vel.v0 = _mm512_set1_ps(epj[j].vel.x);

EPJ_vel.v1 = _mm512_set1_ps(epj[j].vel.y);

EPJ_vel.v2 = _mm512_set1_ps(epj[j].vel.z);

__m512x3 acc_sprg_tmp;

__m512x3 acc_dash_tmp;

__m512x3 acc_grav_tmp;

__m512 pot_sprg_tmp;

__m512 pot_grav_tmp;

__m512x3 rij;

__m512 __fkg_tmp10;

__m512 __fkg_tmp9;

__m512 r_real_sq;

__m512 r_coll_tmp;

__m512 r_coll_sq;

__m512 over_r_real;

__m512 over_r_real_sq;

__m512 r_coll_cu;

__m512 over_r_coll_cu;

__m512 tmp0;

__m512 __fkg_tmp0;

__m512x3 __fkg_tmp4;

__m512 pot_offset;

__m512 __fkg_tmp5;

__m512 m_red;

__m512 r_real;

__m512 dr;

__m512 __fkg_tmp1;

__m512x3 __fkg_tmp6;

__m512 __fkg_tmp7;

__m512x3 vij;

__m512 rv;

__m512 __fkg_tmp2;

__m512x3 __fkg_tmp8;

__m512 m_over_r_real;

__m512 tmp1;

__m512 __fkg_tmp3;

__m512 __fkg_tmp12;

__m512 __fkg_tmp11;

__m512 __fkg_tmp14;

__m512 __fkg_tmp13;

__m512 __fkg_tmp16;

__m512 __fkg_tmp15;

__m512 __fkg_tmp17;

acc_sprg_tmp.v0 = _mm512_set1_ps(0.0f);
acc_sprg_tmp.v1 = _mm512_set1_ps(0.0f);
acc_sprg_tmp.v2 = _mm512_set1_ps(0.0f);
acc_dash_tmp.v0 = _mm512_set1_ps(0.0f);
acc_dash_tmp.v1 = _mm512_set1_ps(0.0f);
acc_dash_tmp.v2 = _mm512_set1_ps(0.0f);
acc_grav_tmp.v0 = _mm512_set1_ps(0.0f);
acc_grav_tmp.v1 = _mm512_set1_ps(0.0f);
acc_grav_tmp.v2 = _mm512_set1_ps(0.0f);
pot_sprg_tmp = _mm512_set1_ps(0.0f);
pot_grav_tmp = _mm512_set1_ps(0.0f);
rij.v0 = _mm512_sub_ps(EPI_pos.v0,EPJ_pos.v0);
rij.v1 = _mm512_sub_ps(EPI_pos.v1,EPJ_pos.v1);
rij.v2 = _mm512_sub_ps(EPI_pos.v2,EPJ_pos.v2);
__fkg_tmp10 = _mm512_fmadd_ps(rij.v0,rij.v0,_mm512_set1_ps(eps2));
__fkg_tmp9 = _mm512_fmadd_ps(rij.v1,rij.v1,__fkg_tmp10);
r_real_sq = _mm512_fmadd_ps(rij.v2,rij.v2,__fkg_tmp9);
{
__mmask16 pg1;
__mmask16 pg0;
pg1 = _mm512_cmp_ps_mask(r_real_sq,_mm512_set1_ps(eps2),_CMP_NEQ_OQ);
pg0 = pg1;

r_coll_tmp = _mm512_add_ps(EPI_r_coll,EPJ_r_coll);
r_coll_sq = _mm512_mul_ps(r_coll_tmp,r_coll_tmp);
over_r_real = rsqrt(r_real_sq);
over_r_real_sq = _mm512_mul_ps(over_r_real,over_r_real);
{
__mmask16 pg3;
__mmask16 pg2;
pg3 = _mm512_cmp_ps_mask(r_real_sq,r_coll_sq,_CMP_LT_OQ);
pg2 = pg3;
pg3 = _kand_mask16(pg3,pg1);

r_coll_cu = _mm512_mul_ps(r_coll_sq,r_coll_tmp);
over_r_coll_cu = _mm512_div_ps(_mm512_set1_ps(1.0f),r_coll_cu);
tmp0 = _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(1.0f)),EPJ_mass);
__fkg_tmp0 = _mm512_mul_ps(tmp0,over_r_coll_cu);
__fkg_tmp4.v0 = _mm512_mul_ps(_mm512_mul_ps(tmp0,over_r_coll_cu),rij.v0);
__fkg_tmp4.v1 = _mm512_mul_ps(_mm512_mul_ps(tmp0,over_r_coll_cu),rij.v1);
__fkg_tmp4.v2 = _mm512_mul_ps(_mm512_mul_ps(tmp0,over_r_coll_cu),rij.v2);
pot_offset = _mm512_div_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(1.5f)),r_coll_tmp);
__fkg_tmp5 = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.25f),EPJ_mass),_mm512_fmadd_ps(r_real_sq,over_r_coll_cu,pot_offset));
m_red = _mm512_div_ps(EPJ_mass,_mm512_add_ps(EPI_mass,EPJ_mass));
r_real = _mm512_mul_ps(r_real_sq,over_r_real);
dr = _mm512_sub_ps(r_coll_tmp,r_real);
__fkg_tmp1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real);
__fkg_tmp6.v0 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real),rij.v0);
__fkg_tmp6.v1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real),rij.v1);
__fkg_tmp6.v2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(kappa),m_red),dr),over_r_real),rij.v2);
__fkg_tmp7 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.25f),_mm512_set1_ps(kappa)),m_red),dr),dr);
vij.v0 = _mm512_sub_ps(EPI_vel.v0,EPJ_vel.v0);
vij.v1 = _mm512_sub_ps(EPI_vel.v1,EPJ_vel.v1);
vij.v2 = _mm512_sub_ps(EPI_vel.v2,EPJ_vel.v2);
rv = _mm512_fmadd_ps(rij.v2,vij.v2,_mm512_fmadd_ps(rij.v0,vij.v0,_mm512_mul_ps(rij.v1,vij.v1)));
__fkg_tmp2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq);
__fkg_tmp8.v0 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq),rij.v0);
__fkg_tmp8.v1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq),rij.v1);
__fkg_tmp8.v2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(eta),m_red),rv),over_r_real_sq),rij.v2);
acc_grav_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v0,__fkg_tmp4.v0);;
acc_grav_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v1,__fkg_tmp4.v1);;
acc_grav_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v2,__fkg_tmp4.v2);;
pot_grav_tmp = _mm512_mask_blend_ps(pg3,pot_grav_tmp,__fkg_tmp5);;
acc_sprg_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_sprg_tmp.v0,__fkg_tmp6.v0);;
acc_sprg_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_sprg_tmp.v1,__fkg_tmp6.v1);;
acc_sprg_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_sprg_tmp.v2,__fkg_tmp6.v2);;
pot_sprg_tmp = _mm512_mask_blend_ps(pg3,pot_sprg_tmp,__fkg_tmp7);;
acc_dash_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_dash_tmp.v0,__fkg_tmp8.v0);;
acc_dash_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_dash_tmp.v1,__fkg_tmp8.v1);;
acc_dash_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_dash_tmp.v2,__fkg_tmp8.v2);;
pg3 = _knot_mask16(pg2);
pg3 = _kand_mask16(pg3,pg1);

m_over_r_real = _mm512_mul_ps(EPJ_mass,over_r_real);
tmp1 = _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(1.0f)),m_over_r_real);
__fkg_tmp3 = _mm512_mul_ps(tmp1,over_r_real_sq);
__fkg_tmp4.v0 = _mm512_mul_ps(_mm512_mul_ps(tmp1,over_r_real_sq),rij.v0);
__fkg_tmp4.v1 = _mm512_mul_ps(_mm512_mul_ps(tmp1,over_r_real_sq),rij.v1);
__fkg_tmp4.v2 = _mm512_mul_ps(_mm512_mul_ps(tmp1,over_r_real_sq),rij.v2);
__fkg_tmp5 = _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps((PIKG::F32)0.0),_mm512_set1_ps(0.5f)),m_over_r_real);
acc_grav_tmp.v0 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v0,__fkg_tmp4.v0);;
acc_grav_tmp.v1 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v1,__fkg_tmp4.v1);;
acc_grav_tmp.v2 = _mm512_mask_blend_ps(pg3,acc_grav_tmp.v2,__fkg_tmp4.v2);;
pot_grav_tmp = _mm512_mask_blend_ps(pg3,pot_grav_tmp,__fkg_tmp5);;
}

}

__fkg_tmp12 = _mm512_add_ps(acc_grav_tmp.v0,acc_sprg_tmp.v0);
__fkg_tmp11 = _mm512_add_ps(__fkg_tmp12,acc_dash_tmp.v0);
FORCE_acc.v0 = _mm512_add_ps(FORCE_acc.v0,__fkg_tmp11);
__fkg_tmp14 = _mm512_add_ps(acc_grav_tmp.v1,acc_sprg_tmp.v1);
__fkg_tmp13 = _mm512_add_ps(__fkg_tmp14,acc_dash_tmp.v1);
FORCE_acc.v1 = _mm512_add_ps(FORCE_acc.v1,__fkg_tmp13);
__fkg_tmp16 = _mm512_add_ps(acc_grav_tmp.v2,acc_sprg_tmp.v2);
__fkg_tmp15 = _mm512_add_ps(__fkg_tmp16,acc_dash_tmp.v2);
FORCE_acc.v2 = _mm512_add_ps(FORCE_acc.v2,__fkg_tmp15);
FORCE_acc_dash.v0 = _mm512_add_ps(FORCE_acc_dash.v0,acc_dash_tmp.v0);
FORCE_acc_dash.v1 = _mm512_add_ps(FORCE_acc_dash.v1,acc_dash_tmp.v1);
FORCE_acc_dash.v2 = _mm512_add_ps(FORCE_acc_dash.v2,acc_dash_tmp.v2);
__fkg_tmp17 = _mm512_add_ps(pot_grav_tmp,pot_sprg_tmp);
FORCE_pot = _mm512_add_ps(FORCE_pot,__fkg_tmp17);
} // loop of j
FORCE_acc.v0 = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp18.v0,FORCE_acc.v0);
FORCE_acc.v1 = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp18.v1,FORCE_acc.v1);
FORCE_acc.v2 = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp18.v2,FORCE_acc.v2);
FORCE_acc_dash.v0 = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp19.v0,FORCE_acc_dash.v0);
FORCE_acc_dash.v1 = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp19.v1,FORCE_acc_dash.v1);
FORCE_acc_dash.v2 = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp19.v2,FORCE_acc_dash.v2);
FORCE_pot = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp20,FORCE_pot);
} // if of j tail loop

((float*)&force[i+0].acc.x)[0] += _mm512_reduce_add_ps(FORCE_acc.v0);

((float*)&force[i+0].acc.y)[0] += _mm512_reduce_add_ps(FORCE_acc.v1);

((float*)&force[i+0].acc.z)[0] += _mm512_reduce_add_ps(FORCE_acc.v2);

((float*)&force[i+0].acc_dash.x)[0] += _mm512_reduce_add_ps(FORCE_acc_dash.v0);

((float*)&force[i+0].acc_dash.y)[0] += _mm512_reduce_add_ps(FORCE_acc_dash.v1);

((float*)&force[i+0].acc_dash.z)[0] += _mm512_reduce_add_ps(FORCE_acc_dash.v2);

((float*)&force[i+0].pot)[0] += _mm512_reduce_add_ps(FORCE_pot);

} // loop of i
{ // tail loop of reference 
for(;i < ni;++i){
PIKG::F32 EPI_mass;

EPI_mass = epi[i+0].mass;
PIKG::F32vec EPI_pos;

EPI_pos.x = epi[i+0].pos.x;
EPI_pos.y = epi[i+0].pos.y;
EPI_pos.z = epi[i+0].pos.z;
PIKG::F32 EPI_r_coll;

EPI_r_coll = epi[i+0].r_coll;
PIKG::F32vec EPI_vel;

EPI_vel.x = epi[i+0].vel.x;
EPI_vel.y = epi[i+0].vel.y;
EPI_vel.z = epi[i+0].vel.z;
PIKG::F32vec FORCE_acc;

FORCE_acc.x = 0.0f;
FORCE_acc.y = 0.0f;
FORCE_acc.z = 0.0f;
PIKG::F32vec FORCE_acc_dash;

FORCE_acc_dash.x = 0.0f;
FORCE_acc_dash.y = 0.0f;
FORCE_acc_dash.z = 0.0f;
PIKG::F32 FORCE_pot;

FORCE_pot = 0.0f;
for(j = 0;j < nj;++j){
PIKG::F32 EPJ_mass;

EPJ_mass = epj[j].mass;
PIKG::F32vec EPJ_pos;

EPJ_pos.x = epj[j].pos.x;
EPJ_pos.y = epj[j].pos.y;
EPJ_pos.z = epj[j].pos.z;
PIKG::F32 EPJ_r_coll;

EPJ_r_coll = epj[j].r_coll;
PIKG::F32vec EPJ_vel;

EPJ_vel.x = epj[j].vel.x;
EPJ_vel.y = epj[j].vel.y;
EPJ_vel.z = epj[j].vel.z;
PIKG::F32vec acc_sprg_tmp;

PIKG::F32vec acc_dash_tmp;

PIKG::F32vec acc_grav_tmp;

PIKG::F32 pot_sprg_tmp;

PIKG::F32 pot_grav_tmp;

PIKG::F32vec rij;

PIKG::F32 __fkg_tmp10;

PIKG::F32 __fkg_tmp9;

PIKG::F32 r_real_sq;

PIKG::F32 r_coll_tmp;

PIKG::F32 r_coll_sq;

PIKG::F32 over_r_real;

PIKG::F32 over_r_real_sq;

PIKG::F32 r_coll_cu;

PIKG::F32 over_r_coll_cu;

PIKG::F32 tmp0;

PIKG::F32 __fkg_tmp0;

PIKG::F32vec __fkg_tmp4;

PIKG::F32 pot_offset;

PIKG::F32 __fkg_tmp5;

PIKG::F32 m_red;

PIKG::F32 r_real;

PIKG::F32 dr;

PIKG::F32 __fkg_tmp1;

PIKG::F32vec __fkg_tmp6;

PIKG::F32 __fkg_tmp7;

PIKG::F32vec vij;

PIKG::F32 rv;

PIKG::F32 __fkg_tmp2;

PIKG::F32vec __fkg_tmp8;

PIKG::F32 m_over_r_real;

PIKG::F32 tmp1;

PIKG::F32 __fkg_tmp3;

PIKG::F32 __fkg_tmp12;

PIKG::F32 __fkg_tmp11;

PIKG::F32 __fkg_tmp14;

PIKG::F32 __fkg_tmp13;

PIKG::F32 __fkg_tmp16;

PIKG::F32 __fkg_tmp15;

PIKG::F32 __fkg_tmp17;

acc_sprg_tmp.x = 0.0f;
acc_sprg_tmp.y = 0.0f;
acc_sprg_tmp.z = 0.0f;
acc_dash_tmp.x = 0.0f;
acc_dash_tmp.y = 0.0f;
acc_dash_tmp.z = 0.0f;
acc_grav_tmp.x = 0.0f;
acc_grav_tmp.y = 0.0f;
acc_grav_tmp.z = 0.0f;
pot_sprg_tmp = 0.0f;
pot_grav_tmp = 0.0f;
rij.x = (EPI_pos.x-EPJ_pos.x);
rij.y = (EPI_pos.y-EPJ_pos.y);
rij.z = (EPI_pos.z-EPJ_pos.z);
__fkg_tmp10 = (rij.x*rij.x+eps2);
__fkg_tmp9 = (rij.y*rij.y+__fkg_tmp10);
r_real_sq = (rij.z*rij.z+__fkg_tmp9);
if((r_real_sq!=eps2)){
r_coll_tmp = (EPI_r_coll+EPJ_r_coll);
r_coll_sq = (r_coll_tmp*r_coll_tmp);
over_r_real = rsqrt(r_real_sq);
over_r_real_sq = (over_r_real*over_r_real);
if((r_coll_sq>r_real_sq)){
r_coll_cu = (r_coll_sq*r_coll_tmp);
over_r_coll_cu = (1.0f/r_coll_cu);
tmp0 = (-(1.0f)*EPJ_mass);
__fkg_tmp0 = (tmp0*over_r_coll_cu);
__fkg_tmp4.x = ((tmp0*over_r_coll_cu)*rij.x);
__fkg_tmp4.y = ((tmp0*over_r_coll_cu)*rij.y);
__fkg_tmp4.z = ((tmp0*over_r_coll_cu)*rij.z);
pot_offset = (-(1.5f)/r_coll_tmp);
__fkg_tmp5 = ((0.25f*EPJ_mass)*(r_real_sq*over_r_coll_cu+pot_offset));
m_red = (EPJ_mass/(EPI_mass+EPJ_mass));
r_real = (r_real_sq*over_r_real);
dr = (r_coll_tmp-r_real);
__fkg_tmp1 = (((kappa*m_red)*dr)*over_r_real);
__fkg_tmp6.x = ((((kappa*m_red)*dr)*over_r_real)*rij.x);
__fkg_tmp6.y = ((((kappa*m_red)*dr)*over_r_real)*rij.y);
__fkg_tmp6.z = ((((kappa*m_red)*dr)*over_r_real)*rij.z);
__fkg_tmp7 = ((((0.25f*kappa)*m_red)*dr)*dr);
vij.x = (EPI_vel.x-EPJ_vel.x);
vij.y = (EPI_vel.y-EPJ_vel.y);
vij.z = (EPI_vel.z-EPJ_vel.z);
rv = (rij.z*vij.z+(rij.x*vij.x+(rij.y*vij.y)));
__fkg_tmp2 = (((eta*m_red)*rv)*over_r_real_sq);
__fkg_tmp8.x = ((((eta*m_red)*rv)*over_r_real_sq)*rij.x);
__fkg_tmp8.y = ((((eta*m_red)*rv)*over_r_real_sq)*rij.y);
__fkg_tmp8.z = ((((eta*m_red)*rv)*over_r_real_sq)*rij.z);
acc_grav_tmp.x = __fkg_tmp4.x;
acc_grav_tmp.y = __fkg_tmp4.y;
acc_grav_tmp.z = __fkg_tmp4.z;
pot_grav_tmp = __fkg_tmp5;
acc_sprg_tmp.x = __fkg_tmp6.x;
acc_sprg_tmp.y = __fkg_tmp6.y;
acc_sprg_tmp.z = __fkg_tmp6.z;
pot_sprg_tmp = __fkg_tmp7;
acc_dash_tmp.x = __fkg_tmp8.x;
acc_dash_tmp.y = __fkg_tmp8.y;
acc_dash_tmp.z = __fkg_tmp8.z;
}else{
m_over_r_real = (EPJ_mass*over_r_real);
tmp1 = (-(1.0f)*m_over_r_real);
__fkg_tmp3 = (tmp1*over_r_real_sq);
__fkg_tmp4.x = ((tmp1*over_r_real_sq)*rij.x);
__fkg_tmp4.y = ((tmp1*over_r_real_sq)*rij.y);
__fkg_tmp4.z = ((tmp1*over_r_real_sq)*rij.z);
__fkg_tmp5 = (-(0.5f)*m_over_r_real);
acc_grav_tmp.x = __fkg_tmp4.x;
acc_grav_tmp.y = __fkg_tmp4.y;
acc_grav_tmp.z = __fkg_tmp4.z;
pot_grav_tmp = __fkg_tmp5;
}
}
__fkg_tmp12 = (acc_grav_tmp.x+acc_sprg_tmp.x);
__fkg_tmp11 = (__fkg_tmp12+acc_dash_tmp.x);
FORCE_acc.x = (FORCE_acc.x+__fkg_tmp11);
__fkg_tmp14 = (acc_grav_tmp.y+acc_sprg_tmp.y);
__fkg_tmp13 = (__fkg_tmp14+acc_dash_tmp.y);
FORCE_acc.y = (FORCE_acc.y+__fkg_tmp13);
__fkg_tmp16 = (acc_grav_tmp.z+acc_sprg_tmp.z);
__fkg_tmp15 = (__fkg_tmp16+acc_dash_tmp.z);
FORCE_acc.z = (FORCE_acc.z+__fkg_tmp15);
FORCE_acc_dash.x = (FORCE_acc_dash.x+acc_dash_tmp.x);
FORCE_acc_dash.y = (FORCE_acc_dash.y+acc_dash_tmp.y);
FORCE_acc_dash.z = (FORCE_acc_dash.z+acc_dash_tmp.z);
__fkg_tmp17 = (pot_grav_tmp+pot_sprg_tmp);
FORCE_pot = (FORCE_pot+__fkg_tmp17);
} // loop of j

force[i+0].acc.x = (force[i+0].acc.x+FORCE_acc.x);
force[i+0].acc.y = (force[i+0].acc.y+FORCE_acc.y);
force[i+0].acc.z = (force[i+0].acc.z+FORCE_acc.z);
force[i+0].acc_dash.x = (force[i+0].acc_dash.x+FORCE_acc_dash.x);
force[i+0].acc_dash.y = (force[i+0].acc_dash.y+FORCE_acc_dash.y);
force[i+0].acc_dash.z = (force[i+0].acc_dash.z+FORCE_acc_dash.z);
force[i+0].pot = (force[i+0].pot+FORCE_pot);
} // loop of i
} // end loop of reference 
} // Kernel_I1_J16 definition 
PIKG::F64 rsqrt(PIKG::F64 op){ return 1.0/std::sqrt(op); }
PIKG::F64 sqrt(PIKG::F64 op){ return std::sqrt(op); }
PIKG::F64 inv(PIKG::F64 op){ return 1.0/op; }
PIKG::F64 max(PIKG::F64 a,PIKG::F64 b){ return std::max(a,b);}
PIKG::F64 min(PIKG::F64 a,PIKG::F64 b){ return std::min(a,b);}
PIKG::F32 rsqrt(PIKG::F32 op){ return 1.f/std::sqrt(op); }
PIKG::F32 sqrt(PIKG::F32 op){ return std::sqrt(op); }
PIKG::F32 inv(PIKG::F32 op){ return 1.f/op; }
PIKG::S64 max(PIKG::S64 a,PIKG::S64 b){ return std::max(a,b);}
PIKG::S64 min(PIKG::S64 a,PIKG::S64 b){ return std::min(a,b);}
PIKG::S32 max(PIKG::S32 a,PIKG::S32 b){ return std::max(a,b);}
PIKG::S32 min(PIKG::S32 a,PIKG::S32 b){ return std::min(a,b);}
PIKG::F64 table(PIKG::F64 tab[],PIKG::S64 i){ return tab[i]; }
PIKG::F32 table(PIKG::F32 tab[],PIKG::S32 i){ return tab[i]; }
PIKG::F64 to_float(PIKG::U64 op){return (PIKG::F64)op;}
PIKG::F32 to_float(PIKG::U32 op){return (PIKG::F32)op;}
PIKG::F64 to_float(PIKG::S64 op){return (PIKG::F64)op;}
PIKG::F32 to_float(PIKG::S32 op){return (PIKG::F32)op;}
PIKG::S64   to_int(PIKG::F64 op){return (PIKG::S64)op;}
PIKG::S32   to_int(PIKG::F32 op){return (PIKG::S32)op;}
PIKG::U64  to_uint(PIKG::F64 op){return (PIKG::U64)op;}
PIKG::U32  to_uint(PIKG::F32 op){return (PIKG::U32)op;}
template<typename T> PIKG::F64 to_f64(const T& op){return (PIKG::F64)op;}
template<typename T> PIKG::F32 to_f32(const T& op){return (PIKG::F32)op;}
template<typename T> PIKG::S64 to_s64(const T& op){return (PIKG::S64)op;}
template<typename T> PIKG::S32 to_s32(const T& op){return (PIKG::S32)op;}
template<typename T> PIKG::U64 to_u64(const T& op){return (PIKG::U64)op;}
template<typename T> PIKG::U32 to_u32(const T& op){return (PIKG::U32)op;}
__m512 rsqrt(__m512 op){
return _mm512_rsqrt14_ps(op);
}
__m512 sqrt(__m512 op){ return _mm512_sqrt_ps(op); }
__m512 inv(__m512 op){
__m512 x1 = _mm512_rcp14_ps(op);
__m512 x2 = _mm512_fnmadd_ps(op,x1,_mm512_set1_ps(2.f));
x2 = _mm512_mul_ps(x2,x1);
__m512 ret = _mm512_fnmadd_ps(op,x2,_mm512_set1_ps(2.f));
ret = _mm512_mul_ps(ret,x2);
return ret;
}
__m512d rsqrt(__m512d op){
__m512d rinv = _mm512_rsqrt14_pd(op);
__m512d h = _mm512_mul_pd(op,rinv);
h = _mm512_fnmadd_pd(h,rinv,_mm512_set1_pd(1.0));
__m512d poly = _mm512_fmadd_pd(h,_mm512_set1_pd(0.375),_mm512_set1_pd(0.5));
poly = _mm512_mul_pd(poly,h);
return _mm512_fmadd_pd(rinv,poly,rinv);
}
__m512d max(__m512d a,__m512d b){ return _mm512_max_pd(a,b);}
__m512d min(__m512d a,__m512d b){ return _mm512_min_pd(a,b);}
__m512  max(__m512  a,__m512  b){ return _mm512_max_ps(a,b);}
__m512  min(__m512  a,__m512  b){ return _mm512_min_ps(a,b);}
__m512i max(__m512i a,__m512i b){ return _mm512_max_epi32(a,b);}
__m512i min(__m512i a,__m512i b){ return _mm512_min_epi32(a,b);}
__m512d table(__m512d tab,__m512i index){ return _mm512_permutexvar_pd(index,tab);}
__m512  table(__m512  tab,__m512i index){ return _mm512_permutexvar_ps(index,tab);}
__m512d to_double(__m512i op){ return _mm512_cvt_roundepi64_pd(op,(_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));}
__m512  to_float(__m512i op){ return _mm512_cvt_roundepi32_ps(op,(_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));}
__m512i  to_long(__m512d op){ return _mm512_cvt_roundpd_epi64(op,(_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));}
__m512i  to_int(__m512  op){ return _mm512_cvt_roundps_epi32(op,(_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));}
__m512i  to_ulong(__m512d op){ return _mm512_cvt_roundpd_epu64(op,(_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));}
__m512i  to_uint(__m512  op){ return _mm512_cvt_roundps_epu32(op,(_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));}
};// kernel functor definition 
