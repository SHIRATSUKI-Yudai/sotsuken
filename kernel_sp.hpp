#include<pikg_vector.hpp>
#include<cmath>
#include<limits>
#include<chrono>

#include <pikg_avx512.hpp>
struct CalcForceEpSpImpl{
PIKG::F32 eps2;
CalcForceEpSpImpl(){}
CalcForceEpSpImpl(PIKG::F32 eps2):eps2(eps2){}
void initialize(PIKG::F32 eps2_){
eps2 = eps2_;
}
int kernel_id = 0;
void operator()(const Epi1* __restrict__ epi,const int ni,const Epj1* __restrict__ epj,const int nj,Force1* __restrict__ force,const int kernel_select = 1){
static_assert(sizeof(Epi1) == 12,"check consistency of EPI member variable definition between PIKG source and original source");
static_assert(sizeof(Epj1) == 16,"check consistency of EPJ member variable definition between PIKG source and original source");
static_assert(sizeof(Force1) == 16,"check consistency of FORCE member variable definition between PIKG source and original source");
if(kernel_select>=0) kernel_id = kernel_select;
if(kernel_id == 0){
std::cout << "ni: " << ni << " nj:" << nj << std::endl;
Force1* force_tmp = new Force1[ni];
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
void Kernel_I16_J1(const Epi1* __restrict__ epi,const PIKG::S32 ni,const Epj1* __restrict__ epj,const PIKG::S32 nj,Force1* __restrict__ force){
PIKG::S32 i;
PIKG::S32 j;
for(i = 0;i < (ni/16)*16;i += 16){
__m512x3 EPI_pos;

alignas(32) int32_t index_gather_load0[16] = {0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45};
__m512i vindex_gather_load0 = _mm512_load_epi32(index_gather_load0);
EPI_pos.v0 = _mm512_i32gather_ps(vindex_gather_load0,((float*)&epi[i+0].pos.x),4);
alignas(32) int32_t index_gather_load1[16] = {0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45};
__m512i vindex_gather_load1 = _mm512_load_epi32(index_gather_load1);
EPI_pos.v1 = _mm512_i32gather_ps(vindex_gather_load1,((float*)&epi[i+0].pos.y),4);
alignas(32) int32_t index_gather_load2[16] = {0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45};
__m512i vindex_gather_load2 = _mm512_load_epi32(index_gather_load2);
EPI_pos.v2 = _mm512_i32gather_ps(vindex_gather_load2,((float*)&epi[i+0].pos.z),4);
__m512x3 FORCE_acc;

FORCE_acc.v0 = _mm512_set1_ps(0.0f);
FORCE_acc.v1 = _mm512_set1_ps(0.0f);
FORCE_acc.v2 = _mm512_set1_ps(0.0f);
__m512 FORCE_pot;

FORCE_pot = _mm512_set1_ps(0.0f);
for(j = 0;j < (nj/1)*1;++j){
__m512 EPJ_mass;

EPJ_mass = _mm512_set1_ps(epj[j].mass);

__m512x3 EPJ_pos;

EPJ_pos.v0 = _mm512_set1_ps(epj[j].pos.x);

EPJ_pos.v1 = _mm512_set1_ps(epj[j].pos.y);

EPJ_pos.v2 = _mm512_set1_ps(epj[j].pos.z);

__m512x3 rij;

__m512 __fkg_tmp1;

__m512 __fkg_tmp0;

__m512 r2;

__m512 over_r;

__m512 over_r_sq;

__m512 m_over_r;

__m512 m_over_r_cu;

rij.v0 = _mm512_sub_ps(EPI_pos.v0,EPJ_pos.v0);
rij.v1 = _mm512_sub_ps(EPI_pos.v1,EPJ_pos.v1);
rij.v2 = _mm512_sub_ps(EPI_pos.v2,EPJ_pos.v2);
__fkg_tmp1 = _mm512_fmadd_ps(rij.v0,rij.v0,_mm512_set1_ps(eps2));
__fkg_tmp0 = _mm512_fmadd_ps(rij.v1,rij.v1,__fkg_tmp1);
r2 = _mm512_fmadd_ps(rij.v2,rij.v2,__fkg_tmp0);
over_r = rsqrt(r2);
over_r_sq = _mm512_mul_ps(over_r,over_r);
m_over_r = _mm512_mul_ps(EPJ_mass,over_r);
m_over_r_cu = _mm512_mul_ps(over_r_sq,m_over_r);
FORCE_acc.v0 = _mm512_fnmadd_ps(m_over_r_cu,rij.v0,FORCE_acc.v0);
FORCE_acc.v1 = _mm512_fnmadd_ps(m_over_r_cu,rij.v1,FORCE_acc.v1);
FORCE_acc.v2 = _mm512_fnmadd_ps(m_over_r_cu,rij.v2,FORCE_acc.v2);
FORCE_pot = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f),m_over_r,FORCE_pot);
} // loop of j

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load3[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_gather_load3 = _mm512_load_epi32(index_gather_load3);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load3,((float*)&force[i+0].acc.x),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_acc.v0);
int32_t index_scatter_store0[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_scatter_store0 = _mm512_load_epi32(index_scatter_store0);
_mm512_i32scatter_ps(((float*)&force[i+0].acc.x),vindex_scatter_store0,__fkg_tmp_accum,4);
}

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load4[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_gather_load4 = _mm512_load_epi32(index_gather_load4);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load4,((float*)&force[i+0].acc.y),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_acc.v1);
int32_t index_scatter_store1[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_scatter_store1 = _mm512_load_epi32(index_scatter_store1);
_mm512_i32scatter_ps(((float*)&force[i+0].acc.y),vindex_scatter_store1,__fkg_tmp_accum,4);
}

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load5[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_gather_load5 = _mm512_load_epi32(index_gather_load5);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load5,((float*)&force[i+0].acc.z),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_acc.v2);
int32_t index_scatter_store2[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_scatter_store2 = _mm512_load_epi32(index_scatter_store2);
_mm512_i32scatter_ps(((float*)&force[i+0].acc.z),vindex_scatter_store2,__fkg_tmp_accum,4);
}

{
__m512 __fkg_tmp_accum;
alignas(32) int32_t index_gather_load6[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_gather_load6 = _mm512_load_epi32(index_gather_load6);
__fkg_tmp_accum = _mm512_i32gather_ps(vindex_gather_load6,((float*)&force[i+0].pot),4);
__fkg_tmp_accum = _mm512_add_ps(__fkg_tmp_accum,FORCE_pot);
int32_t index_scatter_store3[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_scatter_store3 = _mm512_load_epi32(index_scatter_store3);
_mm512_i32scatter_ps(((float*)&force[i+0].pot),vindex_scatter_store3,__fkg_tmp_accum,4);
}

} // loop of i
{ // tail loop of reference 
for(;i < ni;++i){
PIKG::F32vec EPI_pos;

EPI_pos.x = epi[i+0].pos.x;
EPI_pos.y = epi[i+0].pos.y;
EPI_pos.z = epi[i+0].pos.z;
PIKG::F32vec FORCE_acc;

FORCE_acc.x = 0.0f;
FORCE_acc.y = 0.0f;
FORCE_acc.z = 0.0f;
PIKG::F32 FORCE_pot;

FORCE_pot = 0.0f;
for(j = 0;j < nj;++j){
PIKG::F32 EPJ_mass;

EPJ_mass = epj[j].mass;
PIKG::F32vec EPJ_pos;

EPJ_pos.x = epj[j].pos.x;
EPJ_pos.y = epj[j].pos.y;
EPJ_pos.z = epj[j].pos.z;
PIKG::F32vec rij;

PIKG::F32 __fkg_tmp1;

PIKG::F32 __fkg_tmp0;

PIKG::F32 r2;

PIKG::F32 over_r;

PIKG::F32 over_r_sq;

PIKG::F32 m_over_r;

PIKG::F32 m_over_r_cu;

rij.x = (EPI_pos.x-EPJ_pos.x);
rij.y = (EPI_pos.y-EPJ_pos.y);
rij.z = (EPI_pos.z-EPJ_pos.z);
__fkg_tmp1 = (rij.x*rij.x+eps2);
__fkg_tmp0 = (rij.y*rij.y+__fkg_tmp1);
r2 = (rij.z*rij.z+__fkg_tmp0);
over_r = rsqrt(r2);
over_r_sq = (over_r*over_r);
m_over_r = (EPJ_mass*over_r);
m_over_r_cu = (over_r_sq*m_over_r);
FORCE_acc.x = (FORCE_acc.x - m_over_r_cu*rij.x);
FORCE_acc.y = (FORCE_acc.y - m_over_r_cu*rij.y);
FORCE_acc.z = (FORCE_acc.z - m_over_r_cu*rij.z);
FORCE_pot = (FORCE_pot - 0.5f*m_over_r);
} // loop of j

force[i+0].acc.x = (force[i+0].acc.x+FORCE_acc.x);
force[i+0].acc.y = (force[i+0].acc.y+FORCE_acc.y);
force[i+0].acc.z = (force[i+0].acc.z+FORCE_acc.z);
force[i+0].pot = (force[i+0].pot+FORCE_pot);
} // loop of i
} // end loop of reference 
} // Kernel_I16_J1 definition 
void Kernel_I1_J16(const Epi1* __restrict__ epi,const PIKG::S32 ni,const Epj1* __restrict__ epj,const PIKG::S32 nj,Force1* __restrict__ force){
PIKG::S32 i;
PIKG::S32 j;
for(i = 0;i < (ni/1)*1;++i){
__m512x3 EPI_pos;

EPI_pos.v0 = _mm512_set1_ps(epi[i+0].pos.x);

EPI_pos.v1 = _mm512_set1_ps(epi[i+0].pos.y);

EPI_pos.v2 = _mm512_set1_ps(epi[i+0].pos.z);

__m512x3 FORCE_acc;

FORCE_acc.v0 = _mm512_set1_ps(0.0f);
FORCE_acc.v1 = _mm512_set1_ps(0.0f);
FORCE_acc.v2 = _mm512_set1_ps(0.0f);
__m512 FORCE_pot;

FORCE_pot = _mm512_set1_ps(0.0f);
for(j = 0;j < (nj/16)*16;j += 16){
__m512 EPJ_mass;

alignas(32) int32_t index_gather_load7[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_gather_load7 = _mm512_load_epi32(index_gather_load7);
EPJ_mass = _mm512_i32gather_ps(vindex_gather_load7,((float*)&epj[j].mass),4);
__m512x3 EPJ_pos;

alignas(32) int32_t index_gather_load8[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_gather_load8 = _mm512_load_epi32(index_gather_load8);
EPJ_pos.v0 = _mm512_i32gather_ps(vindex_gather_load8,((float*)&epj[j].pos.x),4);
alignas(32) int32_t index_gather_load9[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_gather_load9 = _mm512_load_epi32(index_gather_load9);
EPJ_pos.v1 = _mm512_i32gather_ps(vindex_gather_load9,((float*)&epj[j].pos.y),4);
alignas(32) int32_t index_gather_load10[16] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60};
__m512i vindex_gather_load10 = _mm512_load_epi32(index_gather_load10);
EPJ_pos.v2 = _mm512_i32gather_ps(vindex_gather_load10,((float*)&epj[j].pos.z),4);
__m512x3 rij;

__m512 __fkg_tmp1;

__m512 __fkg_tmp0;

__m512 r2;

__m512 over_r;

__m512 over_r_sq;

__m512 m_over_r;

__m512 m_over_r_cu;

rij.v0 = _mm512_sub_ps(EPI_pos.v0,EPJ_pos.v0);
rij.v1 = _mm512_sub_ps(EPI_pos.v1,EPJ_pos.v1);
rij.v2 = _mm512_sub_ps(EPI_pos.v2,EPJ_pos.v2);
__fkg_tmp1 = _mm512_fmadd_ps(rij.v0,rij.v0,_mm512_set1_ps(eps2));
__fkg_tmp0 = _mm512_fmadd_ps(rij.v1,rij.v1,__fkg_tmp1);
r2 = _mm512_fmadd_ps(rij.v2,rij.v2,__fkg_tmp0);
over_r = rsqrt(r2);
over_r_sq = _mm512_mul_ps(over_r,over_r);
m_over_r = _mm512_mul_ps(EPJ_mass,over_r);
m_over_r_cu = _mm512_mul_ps(over_r_sq,m_over_r);
FORCE_acc.v0 = _mm512_fnmadd_ps(m_over_r_cu,rij.v0,FORCE_acc.v0);
FORCE_acc.v1 = _mm512_fnmadd_ps(m_over_r_cu,rij.v1,FORCE_acc.v1);
FORCE_acc.v2 = _mm512_fnmadd_ps(m_over_r_cu,rij.v2,FORCE_acc.v2);
FORCE_pot = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f),m_over_r,FORCE_pot);
} // loop of j

if(j<nj){ // tail j loop
__m512x3 __fkg_tmp2;

__fkg_tmp2.v0 = FORCE_acc.v0;
__fkg_tmp2.v1 = FORCE_acc.v1;
__fkg_tmp2.v2 = FORCE_acc.v2;
__m512 __fkg_tmp3;

__fkg_tmp3 = FORCE_pot;
for(;j < nj;++j){
__m512 EPJ_mass;

EPJ_mass = _mm512_set1_ps(epj[j].mass);

__m512x3 EPJ_pos;

EPJ_pos.v0 = _mm512_set1_ps(epj[j].pos.x);

EPJ_pos.v1 = _mm512_set1_ps(epj[j].pos.y);

EPJ_pos.v2 = _mm512_set1_ps(epj[j].pos.z);

__m512x3 rij;

__m512 __fkg_tmp1;

__m512 __fkg_tmp0;

__m512 r2;

__m512 over_r;

__m512 over_r_sq;

__m512 m_over_r;

__m512 m_over_r_cu;

rij.v0 = _mm512_sub_ps(EPI_pos.v0,EPJ_pos.v0);
rij.v1 = _mm512_sub_ps(EPI_pos.v1,EPJ_pos.v1);
rij.v2 = _mm512_sub_ps(EPI_pos.v2,EPJ_pos.v2);
__fkg_tmp1 = _mm512_fmadd_ps(rij.v0,rij.v0,_mm512_set1_ps(eps2));
__fkg_tmp0 = _mm512_fmadd_ps(rij.v1,rij.v1,__fkg_tmp1);
r2 = _mm512_fmadd_ps(rij.v2,rij.v2,__fkg_tmp0);
over_r = rsqrt(r2);
over_r_sq = _mm512_mul_ps(over_r,over_r);
m_over_r = _mm512_mul_ps(EPJ_mass,over_r);
m_over_r_cu = _mm512_mul_ps(over_r_sq,m_over_r);
FORCE_acc.v0 = _mm512_fnmadd_ps(m_over_r_cu,rij.v0,FORCE_acc.v0);
FORCE_acc.v1 = _mm512_fnmadd_ps(m_over_r_cu,rij.v1,FORCE_acc.v1);
FORCE_acc.v2 = _mm512_fnmadd_ps(m_over_r_cu,rij.v2,FORCE_acc.v2);
FORCE_pot = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f),m_over_r,FORCE_pot);
} // loop of j
FORCE_acc.v0 = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp2.v0,FORCE_acc.v0);
FORCE_acc.v1 = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp2.v1,FORCE_acc.v1);
FORCE_acc.v2 = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp2.v2,FORCE_acc.v2);
FORCE_pot = _mm512_mask_blend_ps(_cvtu32_mask16(0b00000001),__fkg_tmp3,FORCE_pot);
} // if of j tail loop

((float*)&force[i+0].acc.x)[0] += _mm512_reduce_add_ps(FORCE_acc.v0);

((float*)&force[i+0].acc.y)[0] += _mm512_reduce_add_ps(FORCE_acc.v1);

((float*)&force[i+0].acc.z)[0] += _mm512_reduce_add_ps(FORCE_acc.v2);

((float*)&force[i+0].pot)[0] += _mm512_reduce_add_ps(FORCE_pot);

} // loop of i
{ // tail loop of reference 
for(;i < ni;++i){
PIKG::F32vec EPI_pos;

EPI_pos.x = epi[i+0].pos.x;
EPI_pos.y = epi[i+0].pos.y;
EPI_pos.z = epi[i+0].pos.z;
PIKG::F32vec FORCE_acc;

FORCE_acc.x = 0.0f;
FORCE_acc.y = 0.0f;
FORCE_acc.z = 0.0f;
PIKG::F32 FORCE_pot;

FORCE_pot = 0.0f;
for(j = 0;j < nj;++j){
PIKG::F32 EPJ_mass;

EPJ_mass = epj[j].mass;
PIKG::F32vec EPJ_pos;

EPJ_pos.x = epj[j].pos.x;
EPJ_pos.y = epj[j].pos.y;
EPJ_pos.z = epj[j].pos.z;
PIKG::F32vec rij;

PIKG::F32 __fkg_tmp1;

PIKG::F32 __fkg_tmp0;

PIKG::F32 r2;

PIKG::F32 over_r;

PIKG::F32 over_r_sq;

PIKG::F32 m_over_r;

PIKG::F32 m_over_r_cu;

rij.x = (EPI_pos.x-EPJ_pos.x);
rij.y = (EPI_pos.y-EPJ_pos.y);
rij.z = (EPI_pos.z-EPJ_pos.z);
__fkg_tmp1 = (rij.x*rij.x+eps2);
__fkg_tmp0 = (rij.y*rij.y+__fkg_tmp1);
r2 = (rij.z*rij.z+__fkg_tmp0);
over_r = rsqrt(r2);
over_r_sq = (over_r*over_r);
m_over_r = (EPJ_mass*over_r);
m_over_r_cu = (over_r_sq*m_over_r);
FORCE_acc.x = (FORCE_acc.x - m_over_r_cu*rij.x);
FORCE_acc.y = (FORCE_acc.y - m_over_r_cu*rij.y);
FORCE_acc.z = (FORCE_acc.z - m_over_r_cu*rij.z);
FORCE_pot = (FORCE_pot - 0.5f*m_over_r);
} // loop of j

force[i+0].acc.x = (force[i+0].acc.x+FORCE_acc.x);
force[i+0].acc.y = (force[i+0].acc.y+FORCE_acc.y);
force[i+0].acc.z = (force[i+0].acc.z+FORCE_acc.z);
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
