#include <ATen/ATen.h>
#include <vector>
#include <omp.h>
#include "aspt.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include<iostream>
#include <chrono> 
#include "assume_aligned.h"
namespace at {
namespace native {

double time_in_mill_now();
double time_in_mill_now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double time_in_mill =
    (tv.tv_sec) * 1000.0 + (tv.tv_usec) / 1000.0;
  return time_in_mill;
}

#define ERR fprintf(stderr, "ERR\n");

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))
#define FTYPE float

constexpr unsigned floorlog2(unsigned x)
{
    return x == 1 ? 0 : 1+floorlog2(x >> 1);
}


#define MFACTOR (32)
#define LOG_MFACTOR (5)
#define BSIZE (1024/1)
#define BF (BSIZE/32)
#define INIT_GRP (10000000)
#define INIT_LIST (-1)
#define THRESHOLD (2)
#define BH (ASpT_block_height)
#define LOG_BH (floorlog2(BH))
#define BW (128*1)
#define MIN_OCC (BW*3/4)
//#define MIN_OCC (1000)
#define SBSIZE (128)
#define SBF (SBSIZE / 32)
#define DBSIZE (1024)
#define DBF (DBSIZE / 32)
#define SPBSIZE (256)
#define SPBF (SPBSIZE / 32)
#define STHRESHOLD (1024/2*1)
#define SSTRIDE (STHRESHOLD / SPBF)
#define _NTHREAD (68)
#define SC_SIZE (2048)

//#define SIM_VALUE

struct v_struct {
	int row, col;
	FTYPE val;
	int grp;
};


InspectorMetadata<FTYPE> inspect(
    float *a, int nr0, int nc                                    //nr0 ->original number of rows in A mat, ne->no. nnz in A
) { 
    //std::cout << "I am inside inspect in aspt.cc" << "\n";
    int NTHREAD=1;
    int ASpT_block_height=128;
    int *row_ptrs = (int*)malloc(sizeof(int)*(nr0+1));
    float *values = (float*)malloc(sizeof(float)*nr0*nc);
    int *col_indices = (int*)malloc(sizeof(int)*nr0*nc);

    int e=0,d=0;
    row_ptrs[0]=0;
    for(int i=0;i<nr0;i++)
    {
        for(int j=0;j<nc;j++)
        {
            if(a[nc*i+j]!=0)
            {
                e++;
                col_indices[d]=j;
                values[d]=a[nc*i+j];
                d++;
            }
            
        }
        row_ptrs[i+1]=e;    
    }

    int ne = row_ptrs[nr0];                          //no. nnz in A
    int *special = NULL;
    int *special2 = NULL;
    int special_p = 0;

    char scr_pad[NTHREAD][SC_SIZE];


    int sflag = 0;
    double avg0[NTHREAD];
    double avg = 0;
    double vari = 0;
    float sp;
    special = (int *)malloc(sizeof(int)*ne);
    special2 = (int *)malloc(sizeof(int)*ne);
    memset(special, 0, sizeof(int)*ne);
    memset(special2, 0, sizeof(int)*ne);

    ne *= (sflag+1);
    //std::cout<<ne<<"\n";
    sp= 1- (float(ne)/(nr0*nc));
    //std::cout<<"sparsity="<<sp<<"\n";
    int nr = CEIL(nr0,BH)*BH;            
    int npanel = CEIL(nr,BH);
    //std::cout<<"npanel: "<<npanel<<"\n";

    int* csr_e0 = col_indices;         //original col indices
    FTYPE* csr_ev0 = values;           //original values

    // Pad out row_ptrs
    int* csr_v = (int *)malloc(sizeof(int)*(nr+1));

    int i = 0;
    for (; i < nr0 + 1; i++) csr_v[i] = row_ptrs[i];
    for (; i < nr + 1; i++)  csr_v[i] = row_ptrs[nr0];

    int* csr_e = (int *)malloc(sizeof(int)*ne);
    FTYPE* csr_ev = (FTYPE *)malloc(sizeof(FTYPE)*ne);

    int* mcsr_cnt = (int *)malloc(sizeof(int)*(npanel+1));
    int* mcsr_chk = (int *)malloc(sizeof(int)*(npanel+1));
    int* mcsr_e = (int *)malloc(sizeof(int)*ne); // reduced later

    memset(mcsr_cnt, 0, sizeof(int)*(npanel+1));
    memset(mcsr_chk, 0, sizeof(int)*(npanel+1));
    memset(mcsr_e, 0, sizeof(int)*ne);

    //std::cout<<"Line 124"<<"\n";

    int bv_size = CEIL(nc, 32);
    //std::cout<< "bv_size "<<bv_size<<"\n";
    unsigned int **bv = (unsigned int **)malloc(sizeof(unsigned int *)*NTHREAD);
    for(int i=0;i<NTHREAD;i++)
        bv[i] = (unsigned int *)malloc(sizeof(unsigned int)*bv_size);
    int **csr_e1 = (int **)malloc(sizeof(int *)*2);
    short **coo = (short **)malloc(sizeof(short *)*2);
    for(int i=0;i<2;i++) {
        csr_e1[i] = (int *)malloc(sizeof(int)*ne);
        coo[i] = (short *)malloc(sizeof(short)*ne);
    }

    // filtering(WILL)
    //memcpy(csr_e1[0], csr_e0, sizeof(int)*ne);
#pragma omp parallel for num_threads(1) schedule(dynamic, 1)
    for(int row_panel=0; row_panel<nr/BH; row_panel++) {
        for(int i=row_panel*BH; i<(row_panel+1)*BH; i++) {
            for(int j=csr_v[i]; j<csr_v[i+1]; j++) {
                csr_e1[0][j] = csr_e0[j];
            }
        }

    }

    //std::cout<<"Line 149"<<"\n";

#pragma omp parallel for num_threads(1) schedule(dynamic, 1)
    for(int row_panel=0; row_panel<nr/BH; row_panel++) {   //looping over row panels
        int tid = omp_get_thread_num();
        int i, j, t_sum=0;

        // coo generate and mcsr_chk
        memset(scr_pad[tid], 0, sizeof(char)*SC_SIZE);
        for(i=row_panel*BH; i<(row_panel+1)*BH; i++) {       //looping within the rows of a row panel
            // std::cout<<csr_v[i];
            // std::cout<<csr_v[i+1];
            for(j=csr_v[i]; j<csr_v[i+1]; j++) {      //this helps to loop till the nnzs in that row in that panel. like 0-3 (3 nnz);3-7(4nnz);7-11(4nnz) 
                coo[0][j] = (i&(BH-1));
                //printf("i: %d, j: %d, coo[0][j]: %d\n", i, j, coo[0][j]);
                int k = (csr_e0[j]&(SC_SIZE-1)); 
                //std::cout<<k<<"\n";     //
                if(scr_pad[tid][k] < THRESHOLD) {
                    if(scr_pad[tid][k] == THRESHOLD - 1) t_sum++;
                    scr_pad[tid][k]++;
                }
            }
        }
        //std::cout<<"Line 168"<<"\n";
        //std::cout<<"MIN_OCC "<< MIN_OCC <<"\n";
        //std:cout<<"t_sum "<<t_sum<<"\n";         //t_sum basically holds the no. of dense columns for that panel.

        //if (t_sum < MIN_OCC) {                   //basically, if the number of dense columns in the entire panel is less than MIN_OCC, treat the panel as a single panel, no need to do any reordering within that panel.
        if (sp>0.85)    {    
            //std::cout<<"I am not doing reorders"<<"\n";
            mcsr_chk[row_panel] = 1;
            mcsr_cnt[row_panel+1] = 1;           //update the panel ptr 
            continue;
        }
    // Reorder the panel if the number of dense columns is more than the threshold.
    //std::cout<<"Line 175"<<"\n";
        // sorting(merge sort)
        int flag = 0;
        for(int stride = 1; stride <= BH/2; stride *= 2, flag=1-flag) {
            for(int pivot = row_panel*BH; pivot < (row_panel+1)*BH; pivot += stride*2) {
                int l1, l2;
                for(i = l1 = csr_v[pivot], l2 = csr_v[pivot+stride]; l1 < csr_v[pivot+stride] && l2 < csr_v[pivot+stride*2]; i++) {
                    if(csr_e1[flag][l1] <= csr_e1[flag][l2]) {
                        coo[1-flag][i] = coo[flag][l1];
                        csr_e1[1-flag][i] = csr_e1[flag][l1++];
                    }
                    else {
                        coo[1-flag][i] = coo[flag][l2];
                        csr_e1[1-flag][i] = csr_e1[flag][l2++];
                    }
                }
                while(l1 < csr_v[pivot+stride]) {
                    coo[1-flag][i] = coo[flag][l1];
                    csr_e1[1-flag][i++] = csr_e1[flag][l1++];
                }
                while(l2 < csr_v[pivot+stride*2]) {
                    coo[1-flag][i] = coo[flag][l2];
                    csr_e1[1-flag][i++] = csr_e1[flag][l2++];
                }
            }
        }
    //std::cout<<"Line 201"<<"\n";

        int weight=1;

        int cq=0, cr=0;

        // dense bit extract (and mcsr_e making)
        for(i=csr_v[row_panel*BH]+1; i<csr_v[(row_panel+1)*BH]; i++) {
            if(csr_e1[flag][i-1] == csr_e1[flag][i]) weight++;
            else {
                if(weight >= THRESHOLD) {
                    cr++;
                } 				//if(cr == BW) { cq++; cr=0;}
                weight = 1;
            }
        }
        //int reminder = (csr_e1[flag][i-1]&31);
        if(weight >= THRESHOLD) {
            cr++;
        } 		//if(cr == BW) { cq++; cr=0; }
// TODO = occ control
        mcsr_cnt[row_panel+1] = CEIL(cr,BW)+1;

    }

////gettimeofday(&tt1, NULL);
    // prefix-sum
    for(int i=1; i<=npanel;i++)
        mcsr_cnt[i] += mcsr_cnt[i-1];
    //mcsr_e[0] = 0;
    mcsr_e[BH * mcsr_cnt[npanel]] = ne;

////gettimeofday(&tt2, NULL);

#pragma omp parallel for num_threads(1) schedule(dynamic, 1)
    for(int row_panel=0; row_panel<nr/BH; row_panel++) {
        int tid = omp_get_thread_num();
        if(mcsr_chk[row_panel] == 0) {
            int i, j;
            int flag = 0;
            int cq=0, cr=0;
            for(int stride = 1; stride <= BH/2; stride*=2, flag=1-flag);
            int base = (mcsr_cnt[row_panel]*BH);
            int mfactor = mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel];
            int weight=1;

            // mcsr_e making
            for(i=csr_v[row_panel*BH]+1; i<csr_v[(row_panel+1)*BH]; i++) {
                if(csr_e1[flag][i-1] == csr_e1[flag][i]) weight++;
                else {
                    int reminder = (csr_e1[flag][i-1]&31);
                    if(weight >= THRESHOLD) {
                        cr++;
                        bv[tid][csr_e1[flag][i-1]>>5] |= (1<<reminder);
                        for(j=i-weight; j<=i-1; j++) {
                            mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
                        }
                    } else {
                        //bv[tid][csr_e1[flag][i-1]>>5] &= (~0 - (1<<reminder));
                        bv[tid][csr_e1[flag][i-1]>>5] &= (0xFFFFFFFF - (1<<reminder));
                    }
                    if(cr == BW) { cq++; cr=0;}
                    weight = 1;
                }
            }

//fprintf(stderr, "inter : %d\n", i);

            int reminder = (csr_e1[flag][i-1]&31);
            if(weight >= THRESHOLD) {
                cr++;
                bv[tid][csr_e1[flag][i-1]>>5] |= (1<<reminder);
                for(j=i-weight; j<=i-1; j++) {
                    mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
                }
            } else {
                bv[tid][csr_e1[flag][i-1]>>5] &= (0xFFFFFFFF - (1<<reminder));
            }
            // reordering
            int delta = mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel];
            int base0 = mcsr_cnt[row_panel]*BH;
            for(i=row_panel*BH; i<(row_panel+1)*BH; i++) {
                int base = base0+(i-row_panel*BH)*delta;
                int dpnt = mcsr_e[base] = csr_v[i];
                for(int j=1;j<delta;j++) {
                    mcsr_e[base+j] += mcsr_e[base+j-1];
                }
                int spnt=mcsr_e[mcsr_cnt[row_panel]*BH + (mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel])*(i - row_panel*BH + 1) - 1];

                avg0[tid] += csr_v[i+1] - spnt;
                for(j=csr_v[i]; j<csr_v[i+1]; j++) {
                    int k = csr_e0[j];
                    if((bv[tid][k>>5]&(1<<(k&31)))) {
                        csr_e[dpnt] = csr_e0[j];
                        csr_ev[dpnt++] = csr_ev0[j];
                    } else {
                        csr_e[spnt] = csr_e0[j];
                        csr_ev[spnt++] = csr_ev0[j];
                    }
                }
            }
        } else {
            int base0 = mcsr_cnt[row_panel]*BH;
            memcpy(&mcsr_e[base0], &csr_v[row_panel*BH], sizeof(int)*(BH));
            // int base0 = mcsr_cnt[row_panel] * BH;
            // for (int r = 0; r < BH; r++) {
            //     // Each row gets two boundaries: start and end.
            //     mcsr_e[base0 + r * 2]     = csr_v[row_panel * BH + r];       // start boundary
            //     mcsr_e[base0 + r * 2 + 1] = csr_v[row_panel * BH + r + 1];     // end boundary
            // }
            avg0[tid] += csr_v[(row_panel+1)*BH] - csr_v[row_panel*BH];
            int bidx = csr_v[row_panel*BH];
            int bseg = csr_v[(row_panel+1)*BH] - bidx;
            memcpy(&csr_e[bidx], &csr_e0[bidx], sizeof(int)*bseg);
            memcpy(&csr_ev[bidx], &csr_ev0[bidx], sizeof(FTYPE)*bseg);

        }
    }


    for(int i=0;i<NTHREAD;i++)
        avg += avg0[i];
    avg /= (double)nr;

////gettimeofday(&tt3, NULL);

    for(int i=0;i<nr;i++) {
        int idx = (mcsr_cnt[i>>LOG_BH])*BH + (mcsr_cnt[(i>>LOG_BH)+1] - mcsr_cnt[i>>LOG_BH])*((i&(BH-1))+1);
        int diff = csr_v[i+1] - mcsr_e[idx-1];
        double r = ((double)diff - avg);
        vari += r * r;

        if(diff >= STHRESHOLD) {
            int pp = (diff) / STHRESHOLD;
            for(int j=0; j<pp; j++) {
                special[special_p] = i;
                special2[special_p] = j * STHRESHOLD;
                special_p++;
            }
        }
    }
    vari /= (double)nr;

    for(int i=0;i<NTHREAD;i++)
        free(bv[i]);
    for(int i=0;i<2;i++) {
        free(csr_e1[i]);
        free(coo[i]);
    }
    free(bv); free(csr_e1); free(coo);
    free(row_ptrs);
    free(values);
    free(col_indices);

    struct InspectorMetadata<FTYPE> meta;
    meta.nThread = NTHREAD;
    meta.npanel = npanel;
    meta.nr = nr;

    meta.mcsr_e = mcsr_e;
    meta.mcsr_cnt = mcsr_cnt;
    meta.mcsr_chk = mcsr_chk;

    meta.row_ptrs_padded = csr_v;
    meta.col_indices_reordered = csr_e;
    meta.values_reordered = csr_ev;

    meta.avg = avg;
    meta.vari = vari;

    meta.special = special;
    meta.special2 = special2;
    meta.special_p = special_p;
    return meta;
}


//**************************************************THE OG best implementation***************************************************************************/*/
void execute(
    const InspectorMetadata<FTYPE>& meta,
    int nr0, int nc, int sc,
    FTYPE* vin, FTYPE* vout
) {
    //std::cout << "I am inside execute in aspt.cc" << "\n";
    int ASpT_block_height=128;

    int NTHREAD = meta.nThread;
    int npanel = meta.npanel;
    int nr = meta.nr;

    int* mcsr_e = meta.mcsr_e;
    int* mcsr_cnt = meta.mcsr_cnt;
    int* mcsr_chk = meta.mcsr_chk;

    int* special = meta.special;
    int* special2 = meta.special2;
    int special_p = meta.special_p;

    int* csr_v = meta.row_ptrs_padded;
    int* csr_e = meta.col_indices_reordered;
    FTYPE* csr_ev = meta.values_reordered;

    double avg = meta.avg;
    double vari = meta.vari;

    csr_v = assume_aligned<64>(csr_v);
    csr_e = assume_aligned<64>(csr_e);
    csr_ev = assume_aligned<64>(csr_ev);
    vin = assume_aligned<64>(vin);
    vout = assume_aligned<64>(vout);
    int max_threads = omp_get_max_threads();
    memset(vout, 0, sizeof(FTYPE) * nr0 * sc);

    if (vari < 5000 * 1 / 1 * 1) {
#pragma omp parallel for 
        for (int row_panel = 0; row_panel < nr / BH; row_panel++) {
            int stride;
            for (stride = 0; stride < mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel] - 1; stride++) {
                for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++) {
                    int dummy = mcsr_cnt[row_panel] * BH +
                                (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
                    int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

                    int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
                    int j;
                    for (j = loc1; j < interm; j += 8) {
                        for (int k = 0; k < sc; k++) {
                            vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k]
                                               + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k]
                                               + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k]
                                               + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k]
                                               + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k]
                                               + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k]
                                               + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k]
                                               + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
                        }
                    }
                    for (; j < loc2; j++) {
                        for (int k = 0; k < sc; k++) {
                            vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
                        }
                    }
                }
            }
            for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++) {
                int dummy = mcsr_cnt[row_panel] * BH +
                            (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
                int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

                int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
                int j;
                for (j = loc1; j < interm; j += 8) {
                    for (int k = 0; k < sc; k++) {
                        vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k]
                                           + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k]
                                           + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k]
                                           + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k]
                                           + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k]
                                           + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k]
                                           + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k]
                                           + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
                    }
                }
                for (; j < loc2; j++) {
                    for (int k = 0; k < sc; k++) {
                        vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
                    }
                }
            }
        }
    } else {
#pragma omp parallel for
        for (int row_panel = 0; row_panel < nr / BH; row_panel++) {
            int stride;
            for (stride = 0; stride < mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel] - 1; stride++) {
                for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++) {
                    int dummy = mcsr_cnt[row_panel] * BH +
                                (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
                    int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

                    int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
                    int j;
                    for (j = loc1; j < interm; j += 8) {
                        for (int k = 0; k < sc; k++) {
                            vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k]
                                               + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k]
                                               + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k]
                                               + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k]
                                               + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k]
                                               + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k]
                                               + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k]
                                               + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
                        }
                    }
                    for (; j < loc2; j++) {
                        for (int k = 0; k < sc; k++) {
                            vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
                        }
                    }
                }
            }
        }
    }
}
//***********************************************************************************************************************************************************/

std::vector<Tensor> aspt_inspect(const Tensor& dense) {
    TORCH_CHECK(dense.dim() == 2, "Expected 2D tensor");
    auto a_contig = dense.contiguous();
    float* a_ptr = a_contig.data_ptr<float>();

    int rows = dense.size(0);
    int cols = dense.size(1);

    auto meta = inspect(a_ptr, rows, cols);

    auto opts_i32 = at::device(kCPU).dtype(kInt);
    auto opts_f32 = at::device(kCPU).dtype(kFloat);

    int nnz = meta.row_ptrs_padded[rows];  // last value of row_ptrs gives total nnz

    Tensor mcsr_e   = from_blob(meta.mcsr_e, {meta.row_ptrs_padded[rows]}, opts_i32).clone();
    Tensor mcsr_cnt = from_blob(meta.mcsr_cnt, {meta.npanel + 1}, opts_i32).clone();
    Tensor mcsr_chk = from_blob(meta.mcsr_chk, {meta.npanel + 1}, opts_i32).clone();
    Tensor row_ptrs = from_blob(meta.row_ptrs_padded, {meta.nr + 1}, opts_i32).clone();
    Tensor col_idx  = from_blob(meta.col_indices_reordered, {meta.row_ptrs_padded[rows]}, opts_i32).clone();
    Tensor values   = from_blob(meta.values_reordered, {meta.row_ptrs_padded[rows]}, opts_f32).clone();
    Tensor special  =  from_blob(meta.special, {meta.row_ptrs_padded[rows]}, opts_i32).clone();
    Tensor special2  =  from_blob(meta.special2, {meta.row_ptrs_padded[rows]}, opts_i32).clone();

    Tensor avg_tensor     = at::tensor({meta.avg}, opts_f32);
    Tensor vari_tensor    = at::tensor({meta.vari}, opts_f32);
    Tensor nThread_tensor = at::tensor({meta.nThread}, opts_i32);
    Tensor npanel_tensor = at::tensor({meta.npanel}, opts_i32);
    Tensor special_p_tensor = at::tensor({meta.special_p}, opts_i32);
    Tensor nr_tensor = at::tensor({meta.nr}, opts_i32);
    //std::cout << "@@@@@@Entered aspt_inspect!!!" << std::endl;

    return {mcsr_e, mcsr_cnt, mcsr_chk, row_ptrs, col_idx, values,special,special2, avg_tensor, vari_tensor, nThread_tensor, npanel_tensor, special_p_tensor, nr_tensor};
}

Tensor aspt_execute(
    const Tensor& mcsr_e,
    const Tensor& mcsr_cnt,
    const Tensor& mcsr_chk,
    const Tensor& row_ptrs,
    const Tensor& col_idx,
    const Tensor& values,
    const Tensor& special,
    const Tensor& special2,
    const Tensor& avg_tensor,
    const Tensor& vari_tensor,
    const Tensor& nThread_tensor,
    const Tensor& npanel_tensor,
    const Tensor& special_p,
    const Tensor& nr_tensor,
    const Tensor& vin
) {
    int nr = row_ptrs.size(0) - 1;
    int sc = vin.size(1);

    Tensor vout = at::zeros({nr, sc}, vin.options());

    InspectorMetadata<float> meta;
    meta.mcsr_e = mcsr_e.data_ptr<int>();
    meta.mcsr_cnt = mcsr_cnt.data_ptr<int>();
    meta.mcsr_chk = mcsr_chk.data_ptr<int>();
    meta.row_ptrs_padded = row_ptrs.data_ptr<int>();
    meta.col_indices_reordered = col_idx.data_ptr<int>();
    meta.values_reordered = values.data_ptr<float>();
    meta.avg = avg_tensor.item<float>();
    meta.vari = vari_tensor.item<float>();
    meta.nThread = nThread_tensor.item<int>();
    meta.npanel = mcsr_cnt.size(0) - 1;
    meta.nr = nr_tensor.item<int>();
    meta.special = special.data_ptr<int>();
    meta.special2 = special2.data_ptr<int>();
    meta.special_p = special_p.item<int>();

    execute(meta, nr, vin.size(0), sc, vin.data_ptr<float>(), vout.data_ptr<float>());
    //std::cout << "@@@@@@Entered aspt_execute!!!" << std::endl;
    return vout;
}

} // namespace native
} // namespace at
