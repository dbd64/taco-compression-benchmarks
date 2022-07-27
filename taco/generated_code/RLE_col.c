#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_LCM(_a,_b) (taco_lcm((_a),(_b)))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse, taco_mode_sparse3 } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  uint8_t*     fill_value;    // tensor fill value
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#endif
int omp_get_thread_num() { return 0; }
int omp_get_max_threads() { return 1; }
int cmp(const void *a, const void *b) {
  return *((const int*)a) - *((const int*)b);
}
int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target) {
    return arrayStart;
  }
  int lowerBound = arrayStart; // always < target
  int upperBound = arrayEnd; // always >= target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return upperBound;
}
int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}
taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                                  int32_t* dimensions, int32_t* mode_ordering,
                                  taco_mode_t* mode_types) {
  taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));
  t->order         = order;
  t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));
  t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));
  t->csize         = csize;
  for (int32_t i = 0; i < order; i++) {
    t->dimensions[i]    = dimensions[i];
    t->mode_ordering[i] = mode_ordering[i];
    t->mode_types[i]    = mode_types[i];
    switch (t->mode_types[i]) {
      case taco_mode_dense:
        t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));
        break;
      case taco_mode_sparse:
        t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));
        break;
      case taco_mode_sparse3:
        t->indices[i] = (uint8_t **) malloc(3 * sizeof(uint8_t **));
        break;
    }
  }
  return t;
}
void deinit_taco_tensor_t(taco_tensor_t* t) {
  for (int i = 0; i < t->order; i++) {
    free(t->indices[i]);
  }
  free(t->indices);
  free(t->dimensions);
  free(t->mode_ordering);
  free(t->mode_types);
  free(t);
}
unsigned int taco_gcd(unsigned int u, unsigned int v) {
  // TODO: https://lemire.me/blog/2013/12/26/fastest-way-to-compute-the-greatest-common-divisor/
  int shift;
  if (u == 0)
    return v;
  if (v == 0)
    return u;
  for (shift = 0; ((u | v) & 1) == 0; ++shift) {
    u >>= 1;
    v >>= 1;
  }

  while ((u & 1) == 0)
    u >>= 1;

  do {
    while ((v & 1) == 0)
      v >>= 1;
    if (u > v) {
      unsigned int t = v;
      v = u;
      u = t;
    }
    v = v - u;
  } while (v != 0);
  return u << shift;
}
int taco_lcm(int a, int b) {
  if (a==1) return b;
  if (b==1) return a;
  if (a==b) return a;
  int temp = (int) taco_gcd((unsigned)a, (unsigned)b);
  return temp ? (a / temp * b) : 0;
}
#endif

#define MOD3

#ifdef ORIG
int compute(taco_tensor_t *out, taco_tensor_t *mtx_spgemmrle_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  int32_t* restrict out_vals = (int32_t*)(out->vals);
  int32_t out_fill_value = *((int32_t*)(out->fill_value));
  int mtx_spgemmrle_02_dimension = (int)(mtx_spgemmrle_0->dimensions[1]);
  int* restrict mtx_spgemmrle_02_pos = (int*)(mtx_spgemmrle_0->indices[1][0]);
  int* restrict mtx_spgemmrle_02_crd = (int*)(mtx_spgemmrle_0->indices[1][1]);
  int32_t* restrict mtx_spgemmrle_0_vals = (int32_t*)(mtx_spgemmrle_0->vals);
  int32_t mtx_spgemmrle_0_fill_value = *((int32_t*)(mtx_spgemmrle_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  int32_t* restrict vec_rand_vals = (int32_t*)(vec_rand->vals);







  int32_t out_capacity = out1_dimension;
  out_vals = (int32_t*)malloc(sizeof(int32_t) * out_capacity);

  #pragma omp parallel for schedule(static)
  for (int32_t pout = 0; pout < out_capacity; pout++) {
    out_vals[pout] = out_fill_value;
  }



  for (int32_t j_crd = 0; j_crd < vec_rand1_dimension; j_crd++) {
    int32_t i_crd = 0;
    int32_t imtx_spgemmrle_0_pos = mtx_spgemmrle_02_pos[j_crd];
    int32_t pmtx_spgemmrle_02_end = mtx_spgemmrle_02_pos[(j_crd + 1)];

    while (imtx_spgemmrle_0_pos < pmtx_spgemmrle_02_end) {
      int32_t imtx_spgemmrle_0_crd = mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos];
      if (mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos] == i_crd) {
        mtx_spgemmrle_0_fill_value = (&(mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos]))[0];
      }
      if (imtx_spgemmrle_0_crd == i_crd) {
        out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos] * vec_rand_vals[j_crd];
      }
      else {
        if (mtx_spgemmrle_0_fill_value == 0) {
          i_crd = imtx_spgemmrle_0_crd;
          continue;
        }
        else {
          for (int32_t fv = i_crd; fv < imtx_spgemmrle_0_crd; fv++) {
            i_crd = fv;
            out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_fill_value * vec_rand_vals[j_crd];
          }
          i_crd = imtx_spgemmrle_0_crd;
          continue;
        }
      }
      imtx_spgemmrle_0_pos += (int32_t)(imtx_spgemmrle_0_crd == i_crd);
      i_crd++;
    }


    // int32_t imtx_spgemmrle_0_crd = mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos];
    // while (imtx_spgemmrle_0_pos < pmtx_spgemmrle_02_end) {
    //   int li_hoist = vec_rand_vals[j_crd];
    //   while(imtx_spgemmrle_0_crd == i_crd){
    //     out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos] * li_hoist;
    //     i_crd++;
    //     imtx_spgemmrle_0_pos++;
    //     imtx_spgemmrle_0_crd = mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos];
    //   }
    //   mtx_spgemmrle_0_fill_value = (&(mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos-1]))[0];
    //   if (mtx_spgemmrle_0_fill_value == 0) {
    //     i_crd = imtx_spgemmrle_0_crd;
    //   }
    //   else {
    //     int li_hoist = mtx_spgemmrle_0_fill_value * vec_rand_vals[j_crd];
    //     while (i_crd < imtx_spgemmrle_0_crd) {
    //       out_vals[i_crd] = out_vals[i_crd] + li_hoist;
    //       i_crd++;
    //     }
    //   }
    // }

    
    while (i_crd < mtx_spgemmrle_02_dimension && i_crd >= 0) {
      out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_fill_value * vec_rand_vals[j_crd];
      i_crd++;
    }
  }

  out->vals = (uint8_t*)out_vals;
  return 0;
}
#endif

#ifdef MOD1 //2.65
int compute(taco_tensor_t *out, taco_tensor_t *mtx_spgemmrle_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  int32_t* restrict out_vals = (int32_t*)(out->vals);
  int32_t out_fill_value = *((int32_t*)(out->fill_value));
  int mtx_spgemmrle_02_dimension = (int)(mtx_spgemmrle_0->dimensions[1]);
  int* restrict mtx_spgemmrle_02_pos = (int*)(mtx_spgemmrle_0->indices[1][0]);
  int* restrict mtx_spgemmrle_02_crd = (int*)(mtx_spgemmrle_0->indices[1][1]);
  int32_t* restrict mtx_spgemmrle_0_vals = (int32_t*)(mtx_spgemmrle_0->vals);
  int32_t mtx_spgemmrle_0_fill_value = *((int32_t*)(mtx_spgemmrle_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  int32_t* restrict vec_rand_vals = (int32_t*)(vec_rand->vals);







  int32_t out_capacity = out1_dimension;
  out_vals = (int32_t*)malloc(sizeof(int32_t) * out_capacity);

  #pragma omp parallel for schedule(static)
  for (int32_t pout = 0; pout < out_capacity; pout++) {
    out_vals[pout] = out_fill_value;
  }



  for (int32_t j_crd = 0; j_crd < vec_rand1_dimension; j_crd++) {
    int32_t i_crd = 0;
    int32_t imtx_spgemmrle_0_pos = mtx_spgemmrle_02_pos[j_crd];
    int32_t pmtx_spgemmrle_02_end = mtx_spgemmrle_02_pos[(j_crd + 1)];

    int hoist0 = vec_rand_vals[j_crd];
    while (imtx_spgemmrle_0_pos < pmtx_spgemmrle_02_end) {
      int32_t imtx_spgemmrle_0_crd = mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos];
      if (mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos] == i_crd) {
        mtx_spgemmrle_0_fill_value = (&(mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos]))[0];
      }
      if (imtx_spgemmrle_0_crd == i_crd) {
        out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos] * hoist0;
      }
      else {
        if (mtx_spgemmrle_0_fill_value == 0) {
          i_crd = imtx_spgemmrle_0_crd;
          continue;
        }
        else {
          int hoist1 = mtx_spgemmrle_0_fill_value * hoist0;
          for (int32_t fv = i_crd; fv < imtx_spgemmrle_0_crd; fv++) {
            i_crd = fv;
            out_vals[i_crd] = out_vals[i_crd] + hoist1;
          }
          i_crd = imtx_spgemmrle_0_crd;
          continue;
        }
      }
      imtx_spgemmrle_0_pos += (int32_t)(imtx_spgemmrle_0_crd == i_crd);
      i_crd++;
    }
    
    int hoist2 = mtx_spgemmrle_0_fill_value * vec_rand_vals[j_crd];
    while (i_crd < mtx_spgemmrle_02_dimension) {
      out_vals[i_crd] = out_vals[i_crd] + hoist2;
      i_crd++;
    }
  }

  out->vals = (uint8_t*)out_vals;
  return 0;
}
#endif

#ifdef MOD2 // 2.64
int compute(taco_tensor_t *out, taco_tensor_t *mtx_spgemmrle_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  int32_t* restrict out_vals = (int32_t*)(out->vals);
  int32_t out_fill_value = *((int32_t*)(out->fill_value));
  int mtx_spgemmrle_02_dimension = (int)(mtx_spgemmrle_0->dimensions[1]);
  int* restrict mtx_spgemmrle_02_pos = (int*)(mtx_spgemmrle_0->indices[1][0]);
  int* restrict mtx_spgemmrle_02_crd = (int*)(mtx_spgemmrle_0->indices[1][1]);
  int32_t* restrict mtx_spgemmrle_0_vals = (int32_t*)(mtx_spgemmrle_0->vals);
  int32_t mtx_spgemmrle_0_fill_value = *((int32_t*)(mtx_spgemmrle_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  int32_t* restrict vec_rand_vals = (int32_t*)(vec_rand->vals);

  int32_t out_capacity = out1_dimension;
  out_vals = (int32_t*)malloc(sizeof(int32_t) * out_capacity);

  #pragma omp parallel for schedule(static)
  for (int32_t pout = 0; pout < out_capacity; pout++) {
    out_vals[pout] = out_fill_value;
  }

  for (int32_t j_crd = 0; j_crd < vec_rand1_dimension; j_crd++) {
    int32_t i_crd = 0;
    int32_t imtx_spgemmrle_0_pos = mtx_spgemmrle_02_pos[j_crd];
    int32_t pmtx_spgemmrle_02_end = mtx_spgemmrle_02_pos[(j_crd + 1)];

    const int hoist0 = vec_rand_vals[j_crd];
    while (imtx_spgemmrle_0_pos < pmtx_spgemmrle_02_end) {
      int32_t imtx_spgemmrle_0_crd = mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos];
      if (imtx_spgemmrle_0_crd == i_crd) {
        mtx_spgemmrle_0_fill_value = (&(mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos]))[0];
        out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos] * hoist0;
        imtx_spgemmrle_0_pos++;
        i_crd++;
      }
      else {
        if (mtx_spgemmrle_0_fill_value == 0) {
          i_crd = imtx_spgemmrle_0_crd;
        }
        else {
          const int hoist1 = mtx_spgemmrle_0_fill_value * hoist0;
          while(i_crd < imtx_spgemmrle_0_crd){
            out_vals[i_crd] = out_vals[i_crd] + hoist1;
            i_crd++;
          }
        }
      }      
    }
    
    const int hoist2 = mtx_spgemmrle_0_fill_value * vec_rand_vals[j_crd];
    while (i_crd < mtx_spgemmrle_02_dimension) {
      out_vals[i_crd] = out_vals[i_crd] + hoist2;
      i_crd++;
    }
  }

  out->vals = (uint8_t*)out_vals;
  return 0;
}
#endif

#ifdef MOD3 // 2.23 - 19.95
int compute(taco_tensor_t *out, taco_tensor_t *mtx_spgemmrle_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  int32_t* restrict out_vals = (int32_t*)(out->vals);
  int32_t out_fill_value = *((int32_t*)(out->fill_value));
  int mtx_spgemmrle_02_dimension = (int)(mtx_spgemmrle_0->dimensions[1]);
  int* restrict mtx_spgemmrle_02_pos = (int*)(mtx_spgemmrle_0->indices[1][0]);
  int* restrict mtx_spgemmrle_02_crd = (int*)(mtx_spgemmrle_0->indices[1][1]);
  int32_t* restrict mtx_spgemmrle_0_vals = (int32_t*)(mtx_spgemmrle_0->vals);
  int32_t mtx_spgemmrle_0_fill_value = *((int32_t*)(mtx_spgemmrle_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  int32_t* restrict vec_rand_vals = (int32_t*)(vec_rand->vals);

  int32_t out_capacity = out1_dimension;
  out_vals = (int32_t*)malloc(sizeof(int32_t) * out_capacity);

  #pragma omp parallel for schedule(static)
  for (int32_t pout = 0; pout < out_capacity; pout++) {
    out_vals[pout] = out_fill_value;
  }

  for (int32_t j_crd = 0; j_crd < vec_rand1_dimension; j_crd++) {
    int32_t i_crd = 0;
    int32_t imtx_spgemmrle_0_pos = mtx_spgemmrle_02_pos[j_crd];
    int32_t pmtx_spgemmrle_02_end = mtx_spgemmrle_02_pos[(j_crd + 1)];

    //for (p = pos_start; p < pos_end; ++p) {
    //  int c = crd[p];
    //  int e = crd[p + 1];
    //  int fill = val[p];
    //  for (int i = c; i < e; ++i) {
    //    out[i] += fill * vec[j];
    //  }
    //}

    const int hoist0 = vec_rand_vals[j_crd];

    while (imtx_spgemmrle_0_pos < pmtx_spgemmrle_02_end - 1) {
      // int32_t imtx_spgemmrle_0_crd = mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos];
      int32_t imtx_spgemmrle_0_crd_next = mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos + 1];
      int mtx_spgemmrle_0_fill_value = mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos];
      // if (i_crd + 1 == imtx_spgemmrle_0_crd_next){
      //   out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_fill_value * hoist0;
      //   i_crd++;
      // } 
      // else 
      int hoist1 = mtx_spgemmrle_0_fill_value * hoist0;
      if (hoist1 == 0) {
        i_crd = imtx_spgemmrle_0_crd_next;
      } 
      // else if (mtx_spgemmrle_0_fill_value == 1){
      //   while(i_crd < imtx_spgemmrle_0_crd_next){
      //     out_vals[i_crd] = out_vals[i_crd] + hoist0;
      //     i_crd++;
      //   }
      // }
      else {
        // if (mtx_spgemmrle_0_fill_value == 1) hoist1 = hoist0;
        // else if (hoist0 == 1) hoist1 = mtx_spgemmrle_0_fill_value;
        // else hoist1 = mtx_spgemmrle_0_fill_value * hoist0;
        while(i_crd < imtx_spgemmrle_0_crd_next){
          out_vals[i_crd] = out_vals[i_crd] + hoist1;
          i_crd++;
        }
      }
      imtx_spgemmrle_0_pos++;
    }
    
    int mtx_spgemmrle_0_fill_value = mtx_spgemmrle_0_vals[pmtx_spgemmrle_02_end-1];
    const int hoist2 = mtx_spgemmrle_0_fill_value * hoist0;
    if (hoist2 != 0) {
      while (i_crd < mtx_spgemmrle_02_dimension) {
        out_vals[i_crd] = out_vals[i_crd] + hoist2;
        i_crd++;
      }
    }
  }

  out->vals = (uint8_t*)out_vals;
  return 0;
}
#endif

#ifndef BLOCK_SIZE 
#define BLOCK_SIZE 2000
#endif

#ifdef SCHED1
//     stmt = stmt.split(j, j0, j1, 4);
//     stmt = stmt.split(i, i0, i1, 2000);
//     stmt = stmt.reorder({j0, i0, j1, i1}); // ORIG: j0, j1, i0,i1
int compute(taco_tensor_t *out, taco_tensor_t *mtx_spgemmrle_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  int32_t* restrict out_vals = (int32_t*)(out->vals);
  int32_t out_fill_value = *((int32_t*)(out->fill_value));
  const int mtx_spgemmrle_02_dimension = (int)(mtx_spgemmrle_0->dimensions[1]);
  int* restrict mtx_spgemmrle_02_pos = (int*)(mtx_spgemmrle_0->indices[1][0]);
  int* restrict mtx_spgemmrle_02_crd = (int*)(mtx_spgemmrle_0->indices[1][1]);
  int32_t* restrict mtx_spgemmrle_0_vals = (int32_t*)(mtx_spgemmrle_0->vals);
  int32_t mtx_spgemmrle_0_fill_value = *((int32_t*)(mtx_spgemmrle_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  int32_t* restrict vec_rand_vals = (int32_t*)(vec_rand->vals);







  int32_t out_capacity = out1_dimension;
  out_vals = (int32_t*)malloc(sizeof(int32_t) * out_capacity);

  #pragma omp parallel for schedule(static)
  for (int32_t pout = 0; pout < out_capacity; pout++) {
    out_vals[pout] = out_fill_value;
  }

  const int blk_sz = BLOCK_SIZE;

  int32_t* col_pos_arr = (int32_t*)malloc(sizeof(int32_t) * vec_rand1_dimension); // Store the position at every column
  int32_t* col_fill_arr = (int32_t*)malloc(sizeof(int32_t) * vec_rand1_dimension); // Store the position at every column
  for (int32_t j = 0; j < vec_rand1_dimension; j++) {
    col_pos_arr[j] = mtx_spgemmrle_02_pos[j];
    col_fill_arr[j] = mtx_spgemmrle_0_fill_value;
  }

  for (int32_t i_blk = 0; i_blk  < ((mtx_spgemmrle_02_dimension + (blk_sz-1))/blk_sz); i_blk++){
    for (int32_t j_crd = 0; j_crd < vec_rand1_dimension; j_crd++) {
      int32_t i_crd = i_blk * blk_sz;
      const int32_t i_end = (i_blk+1) * blk_sz;
      int32_t imtx_spgemmrle_0_pos = col_pos_arr[j_crd];
      int32_t pmtx_spgemmrle_02_end = mtx_spgemmrle_02_pos[(j_crd + 1)];
      mtx_spgemmrle_0_fill_value = col_fill_arr[j_crd];

      while (imtx_spgemmrle_0_pos < pmtx_spgemmrle_02_end && i_crd < i_end) {
        int32_t imtx_spgemmrle_0_crd = mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos];
        if (imtx_spgemmrle_0_crd == i_crd) {
          // printf("[NON-FILL] %d, %d, %d\n", j_crd, i_blk, i_crd);
          mtx_spgemmrle_0_fill_value = (&(mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos]))[0];
          out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos] * vec_rand_vals[j_crd];
          imtx_spgemmrle_0_pos++;
          i_crd++;
        }
        else {
          if (mtx_spgemmrle_0_fill_value == 0) {
            // printf("[FILL 0] %d, %d, %d\n", j_crd, i_blk, i_crd);
            i_crd = TACO_MIN(imtx_spgemmrle_0_crd, i_end);
            continue;
          }
          else {
            for (; i_crd < TACO_MIN(imtx_spgemmrle_0_crd, i_end); i_crd++) {
              // printf("[FILL] %d, %d, %d\n", j_crd, i_blk, i_crd);
              out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_fill_value * vec_rand_vals[j_crd];
            }
            continue;
          }
        }
      }
      while (i_crd < TACO_MIN(mtx_spgemmrle_02_dimension, i_end)) {
        // printf("[CLEANUP] %d, %d, %d\n", j_crd, i_blk, i_crd);
        out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_fill_value * vec_rand_vals[j_crd];
        i_crd++;
      }
      col_pos_arr[j_crd] = imtx_spgemmrle_0_pos;
      col_fill_arr[j_crd] = mtx_spgemmrle_0_fill_value;
    }
  } 

  out->vals = (uint8_t*)out_vals;
  return 0;
}
#endif


#ifdef MOD99
int compute(taco_tensor_t *out, taco_tensor_t *mtx_spgemmrle_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  int32_t* restrict out_vals = (int32_t*)(out->vals);
  int32_t out_fill_value = *((int32_t*)(out->fill_value));
  int mtx_spgemmrle_02_dimension = (int)(mtx_spgemmrle_0->dimensions[1]);
  int* restrict mtx_spgemmrle_02_pos = (int*)(mtx_spgemmrle_0->indices[1][0]);
  int* restrict mtx_spgemmrle_02_crd = (int*)(mtx_spgemmrle_0->indices[1][1]);
  int32_t* restrict mtx_spgemmrle_0_vals = (int32_t*)(mtx_spgemmrle_0->vals);
  int32_t mtx_spgemmrle_0_fill_value = *((int32_t*)(mtx_spgemmrle_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  int32_t* restrict vec_rand_vals = (int32_t*)(vec_rand->vals);







  int32_t out_capacity = out1_dimension;
  out_vals = (int32_t*)malloc(sizeof(int32_t) * out_capacity);

  #pragma omp parallel for schedule(static)
  for (int32_t pout = 0; pout < out_capacity; pout++) {
    out_vals[pout] = out_fill_value;
  }



  for (int32_t j_crd = 0; j_crd < vec_rand1_dimension; j_crd++) {
    int32_t i_crd = 0;
    int32_t imtx_spgemmrle_0_pos = mtx_spgemmrle_02_pos[j_crd];
    int32_t pmtx_spgemmrle_02_end = mtx_spgemmrle_02_pos[(j_crd + 1)];


    int32_t imtx_spgemmrle_0_crd = mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos];
    while (imtx_spgemmrle_0_pos < pmtx_spgemmrle_02_end) {
      int li_hoist = vec_rand_vals[j_crd];
      while(imtx_spgemmrle_0_crd == i_crd){
        out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos] * li_hoist;
        i_crd++;
        imtx_spgemmrle_0_pos++;
        imtx_spgemmrle_0_crd = mtx_spgemmrle_02_crd[imtx_spgemmrle_0_pos];
      }
      mtx_spgemmrle_0_fill_value = (&(mtx_spgemmrle_0_vals[imtx_spgemmrle_0_pos-1]))[0];
      if (mtx_spgemmrle_0_fill_value == 0) {
        i_crd = imtx_spgemmrle_0_crd;
      }
      else {
        int li_hoist = mtx_spgemmrle_0_fill_value * vec_rand_vals[j_crd];
        while (i_crd < imtx_spgemmrle_0_crd) {
          out_vals[i_crd] = out_vals[i_crd] + li_hoist;
          i_crd++;
        }
      }
    }

    
    while (i_crd < mtx_spgemmrle_02_dimension && i_crd >= 0) {
      out_vals[i_crd] = out_vals[i_crd] + mtx_spgemmrle_0_fill_value * vec_rand_vals[j_crd];
      i_crd++;
    }
  }

  out->vals = (uint8_t*)out_vals;
  return 0;
}
#endif

int assemble(taco_tensor_t *out, taco_tensor_t *mtx_mnistrle_0, taco_tensor_t *vec_rand) {
  return 1;
}

