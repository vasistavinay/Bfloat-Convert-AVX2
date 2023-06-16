/*
 * Author : Vinay V Vasista
 * Email  : vasistavinay@gmail.com
 * Date   : 30/05/2019
 */

#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include <time.h>

#include <immintrin.h>  // AVX2

#define log(...)
#define debug(...)

#ifdef VERIFY
#define TIMER(...)
#else
#define TIMER(...) __VA_ARGS__
#endif

typedef int16_t bfloatData_t;

//convOp body --> convert Op
template<typename T> struct FromBFloat;
template<typename T> struct ToBFloatData;

/*!
* Given a floating-point value, returns the bfloat representation.
*/
static inline bfloatData_t float_to_bfloat_immed(float f) {

    uint32_t *value = reinterpret_cast<uint32_t *>(&f);
    // compile-time decision if we do the rounding for all values except NaN & inf
    if ( (*value & 0x7f800000u) != 0x7f800000u ) {
        // Rounding algorithm "round to nearest, tie to even":
        // ramp up the result (bfloat) if the MSB of the value's lowerb
        // half is 1 and any of the following circumstances take place:
        // - the result bfloat's LSB is 1 or
        // - the input's lower 15 bits represent non-zero value
        if ((*value & 0x00008000u) and (*value & 0x00017fffu)){
            *value += 0x00010000u;
        }
    }
    return static_cast<uint16_t>(*value >> 16);
}

template<> struct ToBFloatData<float> {
    static bfloatData_t body(float val) {
        return float_to_bfloat_immed(val);
    }
};

static inline float bfloat_to_float_immed(bfloatData_t b) {
    uint16_t *value = reinterpret_cast<uint16_t *>(&b);
    auto u = static_cast<uint32_t>(*value) << 16;
    float *v = reinterpret_cast<float *>(&u);
    return *v;
}

template<> struct FromBFloat<float> {
    static float body(bfloatData_t bfloat) {
        return bfloat_to_float_immed(bfloat);
    }
};



struct bfloatShape2D_t {
    int rows_;
    int cols_;
};

// Constants for DMA blocking
const int MPUalignBits = 5;
const int MPUalign = (1 << MPUalignBits);

static inline int get_aligned_size(int size, int align) {
    return (size + align-1) & ~(align-1);
}

size_t get_blocked_buffer_size(const bfloatShape2D_t *shape, size_t elementSize) {
    //alignment in elements!!
    int         M            = shape->rows_;
    int         N            = shape->cols_;
    int         Ma, Na;

    Ma = get_aligned_size(M, MPUalign);
    Na = get_aligned_size(N, MPUalign);

    return Na * Ma * elementSize;
}

/**
 * Block the data from inData into the buffer provided by outData, optionally
 * doing type conversion (i.e. float <-> bfloat).
 */
template <typename inType, typename outType, typename convOp> int block_and_convert_op
(
    const inType     *inData,
    const bfloatShape2D_t      *shape,
    const bfloatShape2D_t      *elemStride,
    outType          *outData,
    bool              block,
    bool              useCSEBlocking,
    bool              avx2_enabled
) {
    // XXX: unused (always false) - consider removing it completely
    (void) useCSEBlocking;

    // alignment in elements!!
    int         M            = shape->rows_;
    int         N            = shape->cols_;
    int         Na           = get_aligned_size(N, MPUalign);
    size_t      rowStride    = elemStride->rows_;
    size_t      colStride    = elemStride->cols_;

    int nrofElemPerMpuBlock = MPUalign * MPUalign;
    int nrofMpuBlocksPerRow = Na >> MPUalignBits;
    outType temp_storage_for_one_col[MPUalign];
    int temp_storage_for_one_col_VEC[MPUalign >> 1];
    // = new outType[MPUalign];
    size_t bytes_to_copy = MPUalign*sizeof(outType);
    
    for (int row = 0; row < M; row++) {

        int mpuBlockRowIndexInMpuBlock = row & (MPUalign -1);  // [0..15]
        size_t offset_prefix = MPUalign * mpuBlockRowIndexInMpuBlock;
        size_t mpuBlockNum_prefix = (row >> MPUalignBits) * nrofMpuBlocksPerRow;
        size_t rowIndex = row*rowStride;

        //We iterate over one MPUalign chunk at a time
        //This gives the liberty to not calculate the index multiple times

        for (int mpu_block = 0; mpu_block < nrofMpuBlocksPerRow; mpu_block++){
            int start_offset = (mpu_block)*MPUalign;
            int end_offset = start_offset + MPUalign;
            if(end_offset > N){
                end_offset = N;
                bytes_to_copy = (end_offset - start_offset)*sizeof(outType);
            }
            int copy_index = 0;
            //#pragma omp parallel for schedule(static)
            //#pragma simd
            int mpuBlockColumnIndexInMpuBlock = start_offset & (MPUalign -1);
            size_t mpuBlockNum = mpuBlockNum_prefix + (start_offset >> MPUalignBits);
            size_t offset  = offset_prefix +
                             nrofElemPerMpuBlock * mpuBlockNum +
                             mpuBlockColumnIndexInMpuBlock;
            if(block) {
              if (avx2_enabled) {
                static const __m256i c_0x0         = _mm256_set1_epi32(0x0);
                static const __m256i c_0xffffffffu = _mm256_set1_epi32(0xffffffffu);
                static const __m256i c_0x7f800000u = _mm256_set1_epi32(0x7f800000u);
                static const __m256i c_0x00008000u = _mm256_set1_epi32(0x00008000u);
                static const __m256i c_0x00017fffu = _mm256_set1_epi32(0x00017fffu);
                static const __m256i c_0x00010000u = _mm256_set1_epi32(0x00010000u);
                static const __m256i c_shuffle     = _mm256_set_epi8(
                                15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
                                15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
                for(int col = start_offset; col < end_offset; col += 16) {
                  // load
                  // TODO: currently doing unaligned load - change after aligned alloc is done
                  __m256 f1_VEC = _mm256_loadu_ps(&inData[rowIndex+col]);
                  __m256 f2_VEC = _mm256_loadu_ps(&inData[rowIndex+col+8]);

                  /*
                  // cast
                  __m256i in1_VEC = *(reinterpret_cast<__m256i *>(&f1_VEC));  // [ 7, 6, 5, 4, 3, 2, 1, 0]
                  __m256i in2_VEC = *(reinterpret_cast<__m256i *>(&f2_VEC));  // [15,14,13,12,11,10, 9, 8]
                  // prepare interleaved inputs
                  __m256i shr_VEC = _mm256_srli_epi128(in1_VEC, 4);  // 4 bytes; [ -, 7, 6, 5, -, 3, 2, 1]
                  __m256i shl_VEC = _mm256_slli_epi128(in2_VEC, 4);  // 4 bytes; [14,13,12, -,10, 9, 8, -]
                  __m256i value1_VEC = _mm256_blendv_epi8(in1_VEC, shl_VEC, 0b10101010);  // [14, 6,12, 4,10, 2, 8, 0]
                  __m256i value2_VEC = _mm256_blendv_epi8(in2_VEC, shr_VEC, 0b01010101);  // [15, 7,13, 5,11, 3, 9, 1]
                  */
                  __m256i value1_VEC = *(reinterpret_cast<__m256i *>(&f1_VEC));  // [ 7, 6, 5, 4, 3, 2, 1, 0]
                  __m256i value2_VEC = *(reinterpret_cast<__m256i *>(&f2_VEC));  // [15,14,13,12,11,10, 9, 8]

                  /*
                   * Vector 1
                   */
                  // conditional in the outer if
                  __m256i t1_1_VEC = _mm256_and_si256(value1_VEC, c_0x7f800000u);  // p = (v & 0x7f800000u)
                  __m256i t1_2_VEC = _mm256_cmpeq_epi32(t1_1_VEC, c_0x7f800000u);  // x' = (p == 0x7f800000u)

                  // conditionals in the nested inner if
                  __m256i t1_3_VEC = _mm256_and_si256(value1_VEC, c_0x00008000u);  // q = (v & 0x00008000u)
                  __m256i t1_4_VEC = _mm256_and_si256(value1_VEC, c_0x00017fffu);  // r = (v & 0x00017fffu)
                  __m256i t1_5_VEC = _mm256_cmpeq_epi32(t1_3_VEC, c_0x0);  // y' = (q == 0)
                  __m256i t1_6_VEC = _mm256_cmpeq_epi32(t1_4_VEC, c_0x0);  // z' = (r == 0)

                  // The condition mask we need for `v += 0x00010000u;` is:
                  // m = (x && y && z) = (x' || y' || z')'
                  __m256i t1_7_VEC = _mm256_or_si256(t1_5_VEC, t1_6_VEC);  // y' || z'
                  __m256i t1_8_VEC = _mm256_or_si256(t1_2_VEC, t1_7_VEC);  // x' ||  (y' || z')

                  // add
                  __m256i t1_9_VEC = _mm256_add_epi32(value1_VEC, c_0x00010000u);  // `value + 0x00010000u`
                  // select(A, B, mask) => if (mask[i]==true) : B[i], else : A[i]  (apply conditional add)
                  __m256i t1_10_VEC = _mm256_blendv_epi8(t1_9_VEC, value1_VEC, t1_8_VEC);  // m' = (x' ||  y' || z')
                  // >> 16
                  __m256i t1_11_VEC = _mm256_srli_epi32(t1_10_VEC, 16);

                  /*
                   * Vector 2
                   */
                  // conditional in the outer if
                  __m256i t2_1_VEC = _mm256_and_si256(value2_VEC, c_0x7f800000u);  // p = (v & 0x7f800000u)
                  __m256i t2_2_VEC = _mm256_cmpeq_epi32(t2_1_VEC, c_0x7f800000u);  // x' = (p == 0x7f800000u)

                  // conditionals in the nested inner if
                  __m256i t2_3_VEC = _mm256_and_si256(value2_VEC, c_0x00008000u);  // q = (v & 0x00008000u)
                  __m256i t2_4_VEC = _mm256_and_si256(value2_VEC, c_0x00017fffu);  // r = (v & 0x00017fffu)
                  __m256i t2_5_VEC = _mm256_cmpeq_epi32(t2_3_VEC, c_0x0);  // y' = (q == 0)
                  __m256i t2_6_VEC = _mm256_cmpeq_epi32(t2_4_VEC, c_0x0);  // z' = (r == 0)

                  // The condition mask we need for `v += 0x00010000u;` is:
                  // m = (x && y && z) = (x' || y' || z')'
                  __m256i t2_7_VEC = _mm256_or_si256(t2_5_VEC, t2_6_VEC);  // y' || z'
                  __m256i t2_8_VEC = _mm256_or_si256(t2_2_VEC, t2_7_VEC);  // x' ||  (y' || z')

                  // add
                  __m256i t2_9_VEC = _mm256_add_epi32(value2_VEC, c_0x00010000u);  // `value + 0x00010000u`
                  // select mask => if (true) : B, else : A
                  __m256i t2_10_VEC = _mm256_blendv_epi8(t2_9_VEC, value2_VEC, t2_8_VEC);  // m' = (x' ||  y' || z')
                  // >> 16
                  __m256i t2_11_VEC = _mm256_srli_epi32(t2_10_VEC, 16);

                  /* Store Vector1 and Vector2 */
                  // t1_11_VEC = [ -, 7, -, 6, -, 5, -, 4, -, 3, -, 2, -, 1, -, 0]  // 16 * shorts
                  // t2_11_VEC = [ -,15, -,14, -,13, -,12, -,11, -,10, -, 9, -, 8]
                  // t2_12_VEC = [15, -,14, -,13, -,12, -,11, -,10, -, 9, -, 8, -]
                  __m256i t2_12_VEC = _mm256_bslli_epi128(t2_11_VEC, 2);  // shift left by 2 bytes
                  // t_1_VEC   = [15, 7,14, 6,13, 5,12, 4,11, 3,10, 2, 9, 1, 8, 0]
                  __m256i t_1_VEC = _mm256_or_si256(t1_11_VEC, t2_12_VEC);
                  // shuffle within 128 bit lanes
                  // t_2_VEC   = [15,14,13,12, 7, 6, 5, 4,11,10, 9, 8, 3, 2, 1, 0]
                  __m256i t_2_VEC = _mm256_shuffle_epi8(t_1_VEC, c_shuffle);
                  // shuffle long ints
                  // t_3_VEC   = [15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
                  __m256i t_3_VEC = _mm256_permute4x64_epi64(t_2_VEC, 0b11011000);  // 0b11011000 = select(3, 1, 2, 0)

                  // store
                  _mm256_maskstore_epi32(&temp_storage_for_one_col_VEC[copy_index], c_0xffffffffu, t_3_VEC);
                  copy_index += 8;

                  memcpy(outData+offset, temp_storage_for_one_col_VEC, bytes_to_copy);
                }
              } else {
                for(int col = start_offset; col < end_offset; col++){
                  float f = inData[rowIndex+col];
                  uint32_t *value = reinterpret_cast<uint32_t *>(&f);
                  // compile-time decision if we do the rounding for all values except NaN & inf
                  if ( (*value & 0x7f800000u) != 0x7f800000u ) {
                      // Rounding algorithm "round to nearest, tie to even":
                      // ramp up the result (bfloat) if the MSB of the value's lowerb
                      // half is 1 and any of the following circumstances take place:
                      // - the result bfloat's LSB is 1 or
                      // - the input's lower 15 bits represent non-zero value
                      if ((*value & 0x00008000u) and (*value & 0x00017fffu)){
                          *value += 0x00010000u;
                      }
                  }
                  temp_storage_for_one_col[copy_index] = static_cast<uint16_t>(*value >> 16);
                  copy_index++;

                  memcpy(outData+offset, temp_storage_for_one_col, bytes_to_copy);
                }
              }
            }
            else {
              if (avx2_enabled) {
                for (int col = 0; col < MPUalign; col++) {
                  bfloatData_t b = inData[offset+col];
                  uint16_t *value = reinterpret_cast<uint16_t *>(&b);
                  auto u = static_cast<uint32_t>(*value) << 16;
                  float *v = reinterpret_cast<float *>(&u);
                  temp_storage_for_one_col[copy_index] = *v;
                  copy_index++;
                }
              } else {
                for (int col = 0; col < MPUalign; col++) {
                  bfloatData_t b = inData[offset+col];
                  uint16_t *value = reinterpret_cast<uint16_t *>(&b);
                  auto u = static_cast<uint32_t>(*value) << 16;
                  float *v = reinterpret_cast<float *>(&u);
                  temp_storage_for_one_col[copy_index] = *v;
                  copy_index++;
                }
              }
              memcpy(outData+rowIndex+start_offset, temp_storage_for_one_col, bytes_to_copy);
            }
        }
    }
    /*
    std::cout<<"----------INPUT---------"<<std::endl;
    for (int i = 0; i < M; i++){
        for(int j = 0; j< N; j++){
            std::cout<<inData[i*N + j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"----------OUTPUT---------"<<std::endl;
    for (int i = 0; i < M; i++){
        for(int j = 0; j< N; j++){
            std::cout<<outData[i*N + j]<<" ";
        }
        std::cout<<std::endl;
    }
    */
    
    
    
    return 0;
}

//New code ends

#define EPS 0.0000001

bool is_equal(float *a, float *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float delta = a[i] - b[i];
    if (delta > EPS || delta < EPS)
      return false;
  }
  return true;
}

bool is_equal(bfloatData_t *a, bfloatData_t *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

#ifndef R
#define R 1024
#endif

#ifndef C
#define C 1024
#endif

int main() 
{    
    float     *inData __attribute__((aligned(64)));
    bfloatShape2D_t     *shape;
    bfloatShape2D_t     *elemStride;
    bfloatData_t        *outData  __attribute__((aligned(32)));
    TIMER(clock_t start, end);
    TIMER(double time_used);

    shape = new bfloatShape2D_t;
    elemStride = new bfloatShape2D_t;
    shape->rows_ = R;
    shape->cols_ = C;

    int numelem = shape->rows_ * shape->cols_;
    std::cout <<"shape = " <<shape->rows_ <<"x" <<shape->cols_ <<std::endl;
    std::cout <<"numelem = " <<numelem <<std::endl;

    // NOTE: aligned new[] is supported since c++17
    // inData = new float[numelem];
    // outData = new bfloatData_t[numelem];
    // TODO: aligned load
    inData = static_cast<float *>(malloc(numelem * sizeof(float)));
    outData = static_cast<bfloatData_t *>(malloc(numelem * sizeof(bfloatData_t)));

    int k = 1;
    for (int i = 0; i < shape->rows_; i++){
        for(int j = 0; j < shape->cols_; j++){
            inData[i*shape->cols_ + j] = k;
            k++;
        }
    }

    elemStride->rows_ = shape->cols_;
    elemStride->cols_ = 1;
    int x;

#ifdef AVX2_ON
    bool avx2_enabled = true;
#else
    bool avx2_enabled = false;
#endif

    /*
     * Float -> BFloat
     */
    TIMER(start = clock());
    x = block_and_convert_op<float, bfloatData_t, ToBFloatData<float>>(inData, shape, elemStride, outData,
            true, false, avx2_enabled);
#ifdef AVX2_ON
    #ifdef VERIFY
    bfloatData_t *outData_verify = static_cast<bfloatData_t *>(malloc(sizeof(bfloatData_t) * numelem));
    x = block_and_convert_op<float, bfloatData_t, ToBFloatData<float>>(inData, shape, elemStride, outData_verify,
            true, false, !avx2_enabled);
    assert(is_equal(outData, outData_verify, numelem) &&
                   "Incorrect AVX output : Float->BFloat");
    std::cout <<"[Float->BFloat]: AVX v/s Scalar => Tested OK" <<std::endl;
    #endif
#endif
    TIMER(end = clock());
    TIMER(time_used = ((double) (end - start)) / CLOCKS_PER_SEC);
    TIMER(std::cout << "[Float->BFloat]: Time Taken "
                    << ((avx2_enabled) ? "[AVX2]" : "      ") << " : "
                    << time_used << " s" << std::endl);

    /*
     * BFloat -> Float
     */
#ifdef BACK_CONVERT
    //Now convert back
    float      *outData_2;
    // NOTE: aligned new[] is supported since c++17
    // outData_2 = new float[numelem];
    // TODO: aligned load
    outData_2 = static_cast<float *>(malloc(sizeof(float)*numelem));

    TIMER(start = clock());
    x = block_and_convert_op<bfloatData_t, float, FromBFloat<float>>(outData, shape, elemStride, outData_2,
            false , false, avx2_enabled);
#ifdef AVX2_ON
    #ifdef VERIFY
    float *outData_2_verify = static_cast<float *>(malloc(sizeof(float) * numelem));
    x = block_and_convert_op<bfloatData_t, float, FromBFloat<float>>(outData, shape, elemStride, outData_2_verify,
            false , false, !avx2_enabled);
    assert(is_equal(outData_2, outData_2_verify, numelem) &&
                    "Incorrect AVX output : BFloat->Float");
    #endif
#endif
    TIMER(end = clock());
    TIMER(time_used = ((double) (end - start)) / CLOCKS_PER_SEC);
    TIMER(std::cout << "Time Taken by BFloat->Float "
                    << ((avx2_enabled) ? "[AVX2]" : "      ") << " : "
                    << time_used << " s" << std::endl);

    //assert(is_equal(inData, outData_2, numelem) && "Data Corruption");
#endif

    return 0;
}
