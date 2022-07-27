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

int compute(taco_tensor_t *out, taco_tensor_t *mtx_kddcuplz77_0, taco_tensor_t *vec_rand) {
  int out1_dimension = (int)(out->dimensions[0]);
  float* restrict out_vals = (float*)(out->vals);
  float out_fill_value = *((float*)(out->fill_value));
  int mtx_kddcuplz77_01_dimension = (int)(mtx_kddcuplz77_0->dimensions[0]);
  int mtx_kddcuplz77_02_dimension = (int)(mtx_kddcuplz77_0->dimensions[1]);
  int* restrict mtx_kddcuplz77_02_pos = (int*)(mtx_kddcuplz77_0->indices[1][0]);
  float* restrict mtx_kddcuplz77_0_vals = (float*)(mtx_kddcuplz77_0->vals);
  float mtx_kddcuplz77_0_fill_value = *((float*)(mtx_kddcuplz77_0->fill_value));
  float* mtx_kddcuplz77_0_fill_region = ((float*)(mtx_kddcuplz77_0->fill_value));
  int vec_rand1_dimension = (int)(vec_rand->dimensions[0]);
  float* restrict vec_rand_vals = (float*)(vec_rand->vals);

  int32_t mtx_kddcuplz77_0_fill_len = 1;

  int32_t mtx_kddcuplz77_0_fill_index = 0;





  int32_t out_capacity = out1_dimension;
  out_vals = (float*)malloc(sizeof(float) * out_capacity);

  #pragma omp parallel for schedule(static)
  for (int32_t pout = 0; pout < out_capacity; pout++) {
    out_vals[pout] = out_fill_value;
  }



  for (int32_t j_crd = 0; j_crd < vec_rand1_dimension; j_crd++) {
    int32_t i_crd = 0;
    int32_t mtx_kddcuplz77_02_coord = 0;
    int32_t mtx_kddcuplz77_02_dist = 0;
    int32_t mtx_kddcuplz77_02_run = 0;
    int32_t mtx_kddcuplz77_02_found_cnt = 0;
    int32_t mtx_kddcuplz77_02_pos_coord = mtx_kddcuplz77_02_coord;
    int32_t imtx_kddcuplz77_0_pos = mtx_kddcuplz77_02_pos[j_crd];
    int32_t pmtx_kddcuplz77_02_end = mtx_kddcuplz77_02_pos[(j_crd + 1)];

    int32_t imtx_kddcuplz77_0_count = mtx_kddcuplz77_02_found_cnt;
    int32_t imtx_kddcuplz77_0_crd = 0;
    while (imtx_kddcuplz77_0_pos < pmtx_kddcuplz77_02_end) {
      if (i_crd == imtx_kddcuplz77_0_crd && !(bool)imtx_kddcuplz77_0_count) {
        mtx_kddcuplz77_02_pos_coord = mtx_kddcuplz77_02_coord;
        if ((((uint16_t*)(&(((uint8_t*)mtx_kddcuplz77_0_vals)[imtx_kddcuplz77_0_pos])))[0] >> 15 & 1) == 0) {
          mtx_kddcuplz77_02_found_cnt = ((uint16_t*)(&(((uint8_t*)mtx_kddcuplz77_0_vals)[imtx_kddcuplz77_0_pos])))[0] & 32767;
          imtx_kddcuplz77_0_pos += 2;
          mtx_kddcuplz77_02_coord += mtx_kddcuplz77_02_found_cnt;
        }
        else {
          mtx_kddcuplz77_02_found_cnt = 0;
          mtx_kddcuplz77_02_dist = ((uint16_t*)(&(((uint8_t*)mtx_kddcuplz77_0_vals)[(imtx_kddcuplz77_0_pos + 2)])))[0];
          mtx_kddcuplz77_02_run = ((uint16_t*)(&(((uint8_t*)mtx_kddcuplz77_0_vals)[imtx_kddcuplz77_0_pos])))[0] & 32767;
          imtx_kddcuplz77_0_pos += 4;
          mtx_kddcuplz77_02_coord += mtx_kddcuplz77_02_run;
          mtx_kddcuplz77_02_pos_coord = mtx_kddcuplz77_02_coord;
        }
        if (!(bool)mtx_kddcuplz77_02_found_cnt) {
          mtx_kddcuplz77_0_fill_len = TACO_MIN(mtx_kddcuplz77_02_dist / 4,mtx_kddcuplz77_02_run);
          mtx_kddcuplz77_0_fill_index = (mtx_kddcuplz77_0_fill_len == 1 ? 0 : 1);
          mtx_kddcuplz77_0_fill_region = ((float*)(&(((uint8_t*)mtx_kddcuplz77_0_vals)[((imtx_kddcuplz77_0_pos - 4) - mtx_kddcuplz77_02_dist)])));
          mtx_kddcuplz77_0_fill_value = mtx_kddcuplz77_0_fill_region[0];
        }
        imtx_kddcuplz77_0_crd = mtx_kddcuplz77_02_pos_coord;
        imtx_kddcuplz77_0_count = (int32_t)mtx_kddcuplz77_02_found_cnt;
      }
      if (imtx_kddcuplz77_0_crd == i_crd && mtx_kddcuplz77_02_found_cnt) {
        int32_t for_end = imtx_kddcuplz77_0_count;
        for (int32_t l = 0; l < imtx_kddcuplz77_0_count; l++) {
          out_vals[i_crd] = out_vals[i_crd] + ((float*)(&(((uint8_t*)mtx_kddcuplz77_0_vals)[imtx_kddcuplz77_0_pos])))[0] * vec_rand_vals[j_crd];
          imtx_kddcuplz77_0_pos += 4;
          i_crd++;
        }
        imtx_kddcuplz77_0_count -= imtx_kddcuplz77_0_count;
        imtx_kddcuplz77_0_crd += for_end;
        continue;
      }
      else {
        int32_t for_end0 = imtx_kddcuplz77_0_crd - i_crd;
        if (mtx_kddcuplz77_0_fill_len == 1)
          for (int32_t l0 = 0; l0 < for_end0; l0++) {
            out_vals[i_crd] = out_vals[i_crd] + mtx_kddcuplz77_0_fill_value * vec_rand_vals[j_crd];
            i_crd++;
          }

        else {
          for (int32_t l0 = 0; l0 < for_end0; l0++) {
            out_vals[i_crd] = out_vals[i_crd] + mtx_kddcuplz77_0_fill_value * vec_rand_vals[j_crd];
            if (mtx_kddcuplz77_0_fill_len > 1) {
              mtx_kddcuplz77_0_fill_value = mtx_kddcuplz77_0_fill_region[mtx_kddcuplz77_0_fill_index];
              mtx_kddcuplz77_0_fill_index++;
              if (mtx_kddcuplz77_0_fill_index == mtx_kddcuplz77_0_fill_len) mtx_kddcuplz77_0_fill_index = 0;

            }
            i_crd++;
          }
        }
        continue;
      }
      imtx_kddcuplz77_0_pos += (int32_t)(imtx_kddcuplz77_0_crd == i_crd) * 4;
      imtx_kddcuplz77_0_crd += (int32_t)(imtx_kddcuplz77_0_crd == i_crd);
      i_crd++;
    }
    while (i_crd < mtx_kddcuplz77_02_dimension && i_crd >= 0) {
      out_vals[i_crd] = out_vals[i_crd] + mtx_kddcuplz77_0_fill_value * vec_rand_vals[j_crd];
      if (mtx_kddcuplz77_0_fill_len > 1) {
        mtx_kddcuplz77_0_fill_value = mtx_kddcuplz77_0_fill_region[mtx_kddcuplz77_0_fill_index];
        mtx_kddcuplz77_0_fill_index++;
        if (mtx_kddcuplz77_0_fill_index == mtx_kddcuplz77_0_fill_len) mtx_kddcuplz77_0_fill_index = 0;

      }
      i_crd++;
    }
  }

  out->vals = (uint8_t*)out_vals;
  return 0;
}

int assemble(taco_tensor_t *out, taco_tensor_t *mtx_kddcuprle_0, taco_tensor_t *vec_rand) {
    return 1;
}