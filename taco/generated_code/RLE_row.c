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

#define ORIG

#ifdef ORIG
int compute(taco_tensor_t *out, taco_tensor_t *mtx_mnistrle_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  int32_t* restrict out_vals = (int32_t*)(out->vals);
  int mtx_mnistrle_01_dimension = (int)(mtx_mnistrle_0->dimensions[0]);
  int* restrict mtx_mnistrle_02_pos = (int*)(mtx_mnistrle_0->indices[1][0]);
  int* restrict mtx_mnistrle_02_crd = (int*)(mtx_mnistrle_0->indices[1][1]);
  int32_t* restrict mtx_mnistrle_0_vals = (int32_t*)(mtx_mnistrle_0->vals);
  int32_t mtx_mnistrle_0_fill_value = *((int32_t*)(mtx_mnistrle_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  int32_t* restrict vec_rand_vals = (int32_t*)(vec_rand->vals);







  int32_t out_capacity = out1_dimension;
  out_vals = (int32_t*)malloc(sizeof(int32_t) * out_capacity);



  #pragma omp parallel for schedule(runtime)
  for (int32_t i_crd = 0; i_crd < mtx_mnistrle_01_dimension; i_crd++) {
    int32_t tjout_val = 0;
    int32_t j_crd = 0;
    int32_t jmtx_mnistrle_0_pos = mtx_mnistrle_02_pos[i_crd];
    int32_t pmtx_mnistrle_02_end = mtx_mnistrle_02_pos[(i_crd + 1)];

    while (jmtx_mnistrle_0_pos < pmtx_mnistrle_02_end) {
      int32_t jmtx_mnistrle_0_crd = mtx_mnistrle_02_crd[jmtx_mnistrle_0_pos];
      if (mtx_mnistrle_02_crd[jmtx_mnistrle_0_pos] == j_crd) {
        mtx_mnistrle_0_fill_value = (&(mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos]))[0];
      }
      if (jmtx_mnistrle_0_crd == j_crd) {
        tjout_val += mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos] * vec_rand_vals[j_crd];
      }
      else {
        int32_t acc = 0;
        for (int32_t fv = j_crd; fv < jmtx_mnistrle_0_crd; fv++) {
          acc += vec_rand_vals[fv];
        }
        j_crd = jmtx_mnistrle_0_crd;
        tjout_val += acc * mtx_mnistrle_0_fill_value;
        continue;
      }
      jmtx_mnistrle_0_pos += (int32_t)(jmtx_mnistrle_0_crd == j_crd);
      j_crd++;
    }
    while (j_crd < vec_rand1_dimension && j_crd >= 0) {
      int32_t acc0 = 0;
      for (int32_t fv0 = j_crd; fv0 < vec_rand1_dimension; fv0++) {
        acc0 += vec_rand_vals[fv0];
      }
      j_crd = vec_rand1_dimension;
      tjout_val += acc0 * mtx_mnistrle_0_fill_value;
      continue;
      j_crd++;
    }
    out_vals[i_crd] = tjout_val;
  }

  out->vals = (uint8_t*)out_vals;
  return 0;
}
#endif

#ifdef MOD1
int compute(taco_tensor_t *out, taco_tensor_t *mtx_mnistrle_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  int32_t* restrict out_vals = (int32_t*)(out->vals);
  int mtx_mnistrle_01_dimension = (int)(mtx_mnistrle_0->dimensions[0]);
  int* restrict mtx_mnistrle_02_pos = (int*)(mtx_mnistrle_0->indices[1][0]);
  int* restrict mtx_mnistrle_02_crd = (int*)(mtx_mnistrle_0->indices[1][1]);
  int32_t* restrict mtx_mnistrle_0_vals = (int32_t*)(mtx_mnistrle_0->vals);
  int32_t mtx_mnistrle_0_fill_value = *((int32_t*)(mtx_mnistrle_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  int32_t* restrict vec_rand_vals = (int32_t*)(vec_rand->vals);







  int32_t out_capacity = out1_dimension;
  out_vals = (int32_t*)malloc(sizeof(int32_t) * out_capacity);



  #pragma omp parallel for schedule(runtime)
  for (int32_t i_crd = 0; i_crd < mtx_mnistrle_01_dimension; i_crd++) {
    int32_t tjout_val = 0;
    int32_t j_crd = 0;
    int32_t jmtx_mnistrle_0_pos = mtx_mnistrle_02_pos[i_crd];
    int32_t pmtx_mnistrle_02_end = mtx_mnistrle_02_pos[(i_crd + 1)];

    while (jmtx_mnistrle_0_pos < pmtx_mnistrle_02_end) {
      int32_t jmtx_mnistrle_0_crd = mtx_mnistrle_02_crd[jmtx_mnistrle_0_pos];
      if (mtx_mnistrle_02_crd[jmtx_mnistrle_0_pos] == j_crd) {
        mtx_mnistrle_0_fill_value = (&(mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos]))[0];
      }
      if (jmtx_mnistrle_0_crd == j_crd) {
        tjout_val += mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos] * vec_rand_vals[j_crd];
      }
      else {
        int32_t acc = 0;
        for (int32_t fv = j_crd; fv < jmtx_mnistrle_0_crd; fv++) {
          acc += vec_rand_vals[fv];
        }
        j_crd = jmtx_mnistrle_0_crd;
        tjout_val += acc * mtx_mnistrle_0_fill_value;
        continue;
      }
      jmtx_mnistrle_0_pos += (int32_t)(jmtx_mnistrle_0_crd == j_crd);
      j_crd++;
    }
    int32_t acc0 = 0;
    while(j_crd < vec_rand1_dimension){
      acc0 += vec_rand_vals[j_crd];
      j_crd++;
    }
    tjout_val += acc0 * mtx_mnistrle_0_fill_value;

    // while (j_crd < vec_rand1_dimension) {
    //   break;
    //   j_crd++;
    // }
    out_vals[i_crd] = tjout_val;
  }

  out->vals = (uint8_t*)out_vals;
  return 0;
}
#endif

#ifdef MOD2 // 75.4
int compute(taco_tensor_t *out, taco_tensor_t *mtx_mnistrle_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  int32_t* restrict out_vals = (int32_t*)(out->vals);
  int mtx_mnistrle_01_dimension = (int)(mtx_mnistrle_0->dimensions[0]);
  int* restrict mtx_mnistrle_02_pos = (int*)(mtx_mnistrle_0->indices[1][0]);
  int* restrict mtx_mnistrle_02_crd = (int*)(mtx_mnistrle_0->indices[1][1]);
  int32_t* restrict mtx_mnistrle_0_vals = (int32_t*)(mtx_mnistrle_0->vals);
  int32_t mtx_mnistrle_0_fill_value = *((int32_t*)(mtx_mnistrle_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  int32_t* restrict vec_rand_vals = (int32_t*)(vec_rand->vals);

  int32_t out_capacity = out1_dimension;
  out_vals = (int32_t*)malloc(sizeof(int32_t) * out_capacity);



  #pragma omp parallel for schedule(runtime)
  for (int32_t i_crd = 0; i_crd < mtx_mnistrle_01_dimension; i_crd++) {
    int32_t tjout_val = 0;
    int32_t j_crd = 0;
    int32_t jmtx_mnistrle_0_pos = mtx_mnistrle_02_pos[i_crd];
    int32_t pmtx_mnistrle_02_end = mtx_mnistrle_02_pos[(i_crd + 1)];


    // int32_t jmtx_mnistrle_0_crd = mtx_mnistrle_02_crd[jmtx_mnistrle_0_pos];
    int32_t jmtx_mnistrle_0_crd_next = mtx_mnistrle_02_crd[jmtx_mnistrle_0_pos+1];
    while (jmtx_mnistrle_0_pos < pmtx_mnistrle_02_end-1) {
      if (j_crd+1 == jmtx_mnistrle_0_crd_next){
        tjout_val += vec_rand_vals[j_crd] * mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos];
        j_crd++;
      } else 
      {
        int mtx_mnistrle_0_fill_value = mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos];
        // if (mtx_mnistrle_0_fill_value != 0) 
        {
          int32_t acc0 = 0;
          while(j_crd < jmtx_mnistrle_0_crd_next){
            acc0 += vec_rand_vals[j_crd];
            j_crd++;
          }
          tjout_val += acc0 * mtx_mnistrle_0_fill_value;
        }
      }
      jmtx_mnistrle_0_pos++;
      jmtx_mnistrle_0_crd_next = mtx_mnistrle_02_crd[jmtx_mnistrle_0_pos];
    }
    if (mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos] != 0)
    {
      int32_t acc1 = 0;
      while(j_crd < vec_rand1_dimension){
        acc1 += vec_rand_vals[j_crd];
        j_crd++;
      }
      tjout_val += acc1 * mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos];
    }
    out_vals[i_crd] = tjout_val;
  }


  out->vals = (uint8_t*)out_vals;
  return 0;
}
#endif

#ifdef MOD3 // 75.4
int compute(taco_tensor_t *out, taco_tensor_t *mtx_mnistrle_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  int32_t* restrict out_vals = (int32_t*)(out->vals);
  int mtx_mnistrle_01_dimension = (int)(mtx_mnistrle_0->dimensions[0]);
  int* restrict mtx_mnistrle_02_pos = (int*)(mtx_mnistrle_0->indices[1][0]);
  int* restrict mtx_mnistrle_02_crd = (int*)(mtx_mnistrle_0->indices[1][1]);
  int32_t* restrict mtx_mnistrle_0_vals = (int32_t*)(mtx_mnistrle_0->vals);
  int32_t mtx_mnistrle_0_fill_value = *((int32_t*)(mtx_mnistrle_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  int32_t* restrict vec_rand_vals = (int32_t*)(vec_rand->vals);

  int32_t out_capacity = out1_dimension;
  out_vals = (int32_t*)malloc(sizeof(int32_t) * out_capacity);



  #pragma omp parallel for schedule(runtime)
  for (int32_t i_crd = 0; i_crd < mtx_mnistrle_01_dimension; i_crd++) {
    int32_t tjout_val = 0;
    int32_t j_crd = 0;
    int32_t jmtx_mnistrle_0_pos = mtx_mnistrle_02_pos[i_crd];
    int32_t pmtx_mnistrle_02_end = mtx_mnistrle_02_pos[(i_crd + 1)];


    // int32_t jmtx_mnistrle_0_crd = mtx_mnistrle_02_crd[jmtx_mnistrle_0_pos];
    while (jmtx_mnistrle_0_pos < pmtx_mnistrle_02_end-1) {
      int32_t jmtx_mnistrle_0_crd_next = mtx_mnistrle_02_crd[jmtx_mnistrle_0_pos+1];
      const int32_t mtx_mnistrle_0_fill_value = mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos];
      if (j_crd+1 == jmtx_mnistrle_0_crd_next){
        tjout_val += vec_rand_vals[j_crd] * mtx_mnistrle_0_fill_value;
        j_crd++;
      } 
      // else if (j_crd+2 == jmtx_mnistrle_0_crd_next)
      // {
      //   tjout_val += vec_rand_vals[j_crd] * mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos];
      //   tjout_val += vec_rand_vals[j_crd+1] * mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos];
      //   j_crd += 2;
      // } 
      else 
      {
        // if (mtx_mnistrle_0_fill_value != 0) 
        {
          int32_t acc0 = 0;
          while(j_crd < jmtx_mnistrle_0_crd_next){
            acc0 += vec_rand_vals[j_crd];
            j_crd++;
          }
          tjout_val += acc0 * mtx_mnistrle_0_fill_value;
        }
      }
      jmtx_mnistrle_0_pos++;
    }
    mtx_mnistrle_0_fill_value = mtx_mnistrle_0_vals[jmtx_mnistrle_0_pos];
    if (mtx_mnistrle_0_fill_value != 0)
    {
      int32_t acc1 = 0;
      while(j_crd < vec_rand1_dimension){
        acc1 += vec_rand_vals[j_crd];
        j_crd++;
      }
      tjout_val += acc1 * mtx_mnistrle_0_fill_value;
    }
    out_vals[i_crd] = tjout_val;
  }


  out->vals = (uint8_t*)out_vals;
  return 0;
}
#endif

int assemble(taco_tensor_t *out, taco_tensor_t *mtx_mnistrle_0, taco_tensor_t *vec_rand) {
  return 1;
}
