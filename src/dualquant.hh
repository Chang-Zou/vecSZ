#ifndef DUALQUANT_HH
#define DUALQUANT_HH

#include <cstddef>
#include <math.h>
#include <vector> 
#include <immintrin.h> //avx intrinsics

#include "types.hh"
#include "constants.hh"
#include "utils/padding.hh"
#include "argument_parser/argparse.hh"
#include "/home/changfz/include/xsimd/xsimd.hpp"

namespace xs = xsimd;
struct lorenzo_1d1l_prequantization
{
  template<typename T, typename Tag, typename Arch>
  void operator()(Arch, T* data, double const *const ebs_L4, size_t blk_end, size_t *id, Tag)
  {
    // defines the simd_instruction sets used in this operation.
    using simd_type = xs::batch<float,Arch>;

    // determine how many iteration to run
    std::size_t increment = simd_type::size;
    std::size_t blk_end = blk_end;
    std::size_t maxSimdSize = blk_end - blk_end % increment;

    // defines variable of simd_type.
    simd_type vdata, veb, vres;
    veb = ebs_L4[EBx2_r];

    for(; *id < maxSimdSize; (*id)+=increment)
    {
      vdata = simd_type::load(&data[(*id)],Tag());
      vres = xs::round(xs::mul(vdata,veb));
      xs::store(&data[(*id)],vres,Tag());
    }
    for (*id = maxSimdSize; *id < blk_end; ++(*id)) //Handle leftovers sequentially
        {
          data[(*id)] = round(data[(*id)] * ebs_L4[EBx2_r]);
        } // end prequantization
  }
};


struct lorenzo_1d1l_postquantization
{
  template<typename T, typename Q, typename Tag, typename Arch>
  void operator()(Arch, T* data, T* outlier, T padding, Q* bcode, auto radius, double const *const ebs_L4, size_t blk_end, size_t *id, size_t _idx0, Tag)
  {
    // defines the simd_instruction sets used in this operation.
    using simd_type = xs::batch<float,Arch>;
    using simd_type_int = xs::batch<int,Arch>;
    using simd_bool = xs::batch_bool<float,Arch>;

    // determine how many iteration to run
    std::size_t increment = simd_type::size;
    std::size_t blk_end = blk_end;
    std::size_t maxSimdSize = blk_end - blk_end % increment;

    // defines varialbe of simd_type
    simd_type vpred, veb, vpad, vradius, vposterr, absvposterr, voutlier, current;
    simd_bool vquant;
    vpad = padding;
    vradius = radius;



    for(; *id < maxSimdSize; (*id)+=increment){
      current = simd_type::load(&data[(*id)],Tag());
      if( *id < _idx0 + 1){
        vpred = vpad; // set vpred to vpad 
      }else {
        vpred = simd_type::load(&data[(*id) - 1], Tag()); // set vpred to data[id - 1]
      }
      vposterr =  xs::sub(current,vpred);
      absvposterr = xs::abs(vposterr); // find the absolute value of vposter
      vquant = xs::le(absvposterr,vradius); // absvposterr < vradius  result save into vquant
      auto _code = xs::broadcast_as<simd_type_int>(xs::add(vposterr,vradius));

      voutlier = xs::bitwise_andnot(vquant, current);

      xs::store(&outlier[(*id)],xs::mul(xs::sub(1, vquant),current),Tag());
      xs::store(&bcode[(*id)],xs::mul(vquant,_code),Tag());
      
    }
    
  }
};


namespace vecsz
{
  namespace predictor_quantizer
  {

    template <typename T, typename Q>
    void c_lorenzo_1d1l(T *data,
                        T *outlier,
                        Q *bcode,
                        size_t const *const dims_L16,
                        double const *const ebs_L4,
                        size_t b0,
                        size_t blksz,
                        int vector_reg,
                        struct SZWorkflow szwf,
                        T pad_constant,
                        int pad_type,
                        T* pad_vals,
                        size_t* pad_idx)
    {
      auto radius = static_cast<Q>(dims_L16[RADIUS]);
      size_t _idx0 = b0 * blksz;

      T padding = 0.0;
      if (szwf.block_padding or szwf.edge_padding)
      {
        size_t dims[3] {blksz, 1, 1};
        padding = padding::find_pad_value(&data[_idx0], pad_type, dims, pad_constant);
        pad_vals[(*pad_idx)++] = padding;
      }
      else if (szwf.global_padding)
      {
          padding = pad_vals[0];
      }

      size_t id = _idx0;
      size_t blk_end = _idx0 + blksz;
    
      // prequantization based on any arch. 
      if(id < dims_L16[LEN])
      {
        //Xsimd decides the best architectures that is being supported by this computer
        xs::dispatch<xs::supported_architectures>(lorenzo_1d1l_prequantization{})(data,ebs_L4,blk_end, &id, xs::unaligned_mode());
      }


      // postquantization
      if (id < dims_L16[DIM0])
      {
        // size_t id = _idx0;
        // size_t blk_end = _idx0 + blksz;

        // for (; id < blksz; id++)
        // {
        //   T pred = id < _idx0 + 1 ? padding : data[id - 1];
        //   T posterr = data[id] - pred;
        //   bool quantizable = fabs(posterr) < radius;
        //   Q _code = static_cast<Q>(posterr + radius);
        //   outlier[id] = (1 - quantizable) * data[id]; //OLD CODE
        //   //data[id]         = (1 - quantizable) * data[id];   //NEW CODE
        //   bcode[id] = quantizable * _code;
        // }
      }
      else
      {
        // for (size_t i0 = 0; i0 < blksz; i0++)
        // {
        //   size_t id = _idx0 + i0;
        //   if (id >= dims_L16[DIM0])
        //     continue;
        //   T pred = id < _idx0 + 1 ? padding : data[id - 1];
        //   T posterr = data[id] - pred;
        //   bool quantizable = fabs(posterr) < radius;
        //   Q _code = static_cast<Q>(posterr + radius);
        //   outlier[id] = (1 - quantizable) * data[id]; //OLD CODE
        //   //data[id]         = (1 - quantizable) * data[id];   //NEW CODE
        //   bcode[id] = quantizable * _code;
        // }
      }
    }

    template <typename T, typename Q>
    void c_lorenzo_2d1l(T *data,
                        T *outlier,
                        Q *bcode,
                        size_t const *const dims_L16,
                        double const *const ebs_L4,
                        size_t b0,
                        size_t b1,
                        size_t blksz,
                        int vector_reg,
                        struct SZWorkflow szwf,
                        T pad_constant,
                        int pad_type,
                        T* pad_vals,
                        size_t* pad_idx)
    {
      alignas(32) T _s[blksz + 1][blksz + 1] {0}; // 2D interpretation of data

      T* _sptr;
      if (szwf.block_padding or szwf.global_padding)
      {
        T block_pad_value;
        T* block = padding::fill_2d_block<T>(data, dims_L16, blksz, b0, b1);
        _sptr = padding::block_pad<T>(block, dims_L16[nDIM], pad_type, blksz, pad_constant, ebs_L4[EBx2_r], &block_pad_value);
        if (szwf.block_padding) pad_vals[(*pad_idx)++] = block_pad_value;
        memcpy(_s, _sptr, sizeof(T) * (blksz + 1) * (blksz + 1));
        free(block);
      }
      else if (szwf.edge_padding)
      {
        T* edge_pad_values = (T *)malloc(sizeof(T) * dims_L16[nDIM]);
        size_t pad_dims[3] {(size_t)blksz, (size_t)blksz, 1};
        T* block = padding::fill_2d_block<T>(data, dims_L16, blksz, b0, b1);
        _sptr = padding::edge_pad<T>(block, dims_L16[nDIM], pad_type, pad_dims, blksz, pad_constant, ebs_L4[EBx2_r], &edge_pad_values);
        memcpy(_s, _sptr, sizeof(T) * (blksz + 1) * (blksz + 1));
        for (size_t i = 0; i < dims_L16[nDIM]; i++) pad_vals[(*pad_idx)++] = edge_pad_values[i];
        free(block);
      }
      else memset(_s, 0, (blksz + 1) * (blksz + 1) * sizeof(T));
      auto radius = static_cast<Q>(dims_L16[RADIUS]);

      size_t _idx1 = b1 * blksz;
      size_t _idx0 = b0 * blksz;

      if (_idx1 + blksz < dims_L16[DIM1] and _idx0 + blksz < dims_L16[DIM0])
      { // vectorizable cases
#ifdef AVX512
        __m512 vradius, vebx2, vzero;
        if (vector_reg == 512)
        {
          __m512 vradius = _mm512_set1_ps(radius);
          __m512 vebx2 = _mm512_set1_ps(ebs_L4[EBx2_r]);
          __m512 vzero = _mm512_setzero_ps();
        }
#endif
        __m256 vradius8 = _mm256_set1_ps(radius);
        __m256 vebx2_8 = _mm256_set1_ps(ebs_L4[EBx2_r]);
        __m256i mask = _mm256_set1_epi32(0xFFFFFFFF);

        //prequantization
        for (size_t i1 = 0; i1 < blksz; i1++)
        {
          size_t i0 = 0;
          size_t id = _idx0 + (_idx1 + i1) * dims_L16[DIM0];
          size_t blk_end = _idx0 + (_idx1 + i1) * dims_L16[DIM0] + blksz;
          size_t blk_end16 = (blk_end & ~0xF);
          size_t blk_end8 = (blk_end & ~0x7);

#ifdef AVX512
          if (vector_reg == 512)
          {
            for (; id < blk_end16; id += 16, i0 += 16)
            { //AVX-512
              __m512 vdata = _mm512_loadu_ps(&data[id]);
              __m512 s = _mm512_roundscale_ps(_mm512_mul_ps(vdata, vebx2), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
              _mm512_storeu_ps(&_s[i1 + 1][i0 + 1], s);
            }
          }
#endif
          for (; id < blk_end8; id += 8, i0 += 8)
          { //AVX2
            __m256 vdata = _mm256_loadu_ps(&data[id]);
            __m256 s = _mm256_round_ps(_mm256_mul_ps(vdata, vebx2_8), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            _mm256_storeu_ps(&_s[i1 + 1][i0 + 1], s);
          }
          for (; i0 < blksz; i0++)
          { //Sequential Case
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            size_t id = gi0 + gi1 * dims_L16[DIM0];
            _s[i1 + 1][i0 + 1] = round(data[id] * ebs_L4[EBx2_r]);
          }
#ifdef PFETCH
          __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + blksz], 0, 0);
#elif PF2
          __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 2 * blksz], 0, 0);
#elif PF4
          __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 4 * blksz], 0, 0);
#elif PF8
          __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 8 * blksz], 0, 0);
#elif PF16
          __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 16 * blksz], 0, 0);
#endif
        }

        // postquantization
        for (size_t i1 = 0; i1 < blksz; i1++)
        {

          size_t i0 = 0;
          size_t gi1 = _idx1 + i1;
          size_t id = _idx0 + gi1 * dims_L16[DIM0];
          size_t blk_end = _idx0 + ((_idx1 + i1) * dims_L16[DIM0]) + blksz;
          size_t blk_end16 = (blk_end & ~0xF);
          size_t blk_end8 = (blk_end & ~0x7);
          size_t blk_end4 = (blk_end & ~0x3);

#ifdef AVX512
          if (vector_reg == 512)
          {
            for (; id < blk_end16; id += 16, i0 += 16)
            { //AVX-512

              __m512 prevW = _mm512_loadu_ps(&_s[i1 + 1][i0]);
              __m512 prevN = _mm512_loadu_ps(&_s[i1][i0 + 1]);
              __m512 prevNW = _mm512_loadu_ps(&_s[i1][i0]);
              __m512 current = _mm512_loadu_ps(&_s[i1 + 1][i0 + 1]);

              __m512 vpred = _mm512_add_ps(prevW, _mm512_sub_ps(prevN, prevNW));
              __m512 vposterr = _mm512_sub_ps(current, vpred);
              __mmask16 vquantizable = _mm512_cmp_ps_mask(_mm512_abs_ps(vposterr), vradius, 1);
              __mmask16 vnotquant = _mm512_cmp_ps_mask(vradius, _mm512_abs_ps(vposterr), 1);
              __m512i _code = _mm512_cvtps_epi32(_mm512_add_ps(vposterr, vradius));
              __m512 voutlier = _mm512_mask_blend_ps(vquantizable, current, vzero);
              __m512i vbcode = _mm512_mask_blend_epi32(vquantizable, _mm512_setzero_epi32(), _code);

              _mm512_storeu_ps(&outlier[id], voutlier);
              _mm512_mask_storeu_epi32(&bcode[id], vquantizable, vbcode);
              _mm512_mask_storeu_epi32(&bcode[id], vnotquant, _mm512_setzero_epi32());
            }
          }
#endif
          for (; id < blk_end8; id += 8, i0 += 8)
          { //AVX-256

            __m256 prevW = _mm256_loadu_ps(&_s[i1 + 1][i0]);
            __m256 prevN = _mm256_loadu_ps(&_s[i1][i0 + 1]);
            __m256 prevNW = _mm256_loadu_ps(&_s[i1][i0]);
            __m256 current = _mm256_loadu_ps(&_s[i1 + 1][i0 + 1]);

            __m256 vpred = _mm256_add_ps(prevW, _mm256_sub_ps(prevN, prevNW));
            __m256 vposterr = _mm256_sub_ps(current, vpred);
            __m256 absposterr = _mm256_sqrt_ps(_mm256_mul_ps(vposterr, vposterr));
            __m256 vquant = _mm256_cmp_ps(absposterr, vradius8, 1);
            __m256i _code = _mm256_cvtps_epi32(_mm256_add_ps(vposterr, vradius8));
            __m256 voutlier = _mm256_andnot_ps(vquant, current);
            __m256i vbcode = _mm256_cvtps_epi32(_mm256_and_ps(vquant, _mm256_cvtepi32_ps(_code)));

            _mm256_storeu_ps(&outlier[id], voutlier);
            _mm256_maskstore_epi32(&bcode[id], mask, vbcode);
          }

          for (; i0 < blksz; i0++)
          {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            size_t id = gi0 + gi1 * dims_L16[DIM0];
            T pred = _s[i1 + 1][i0] + _s[i1][i0 + 1] - _s[i1][i0];
            T posterr = _s[i1 + 1][i0 + 1] - pred;
            bool quantizable = fabs(posterr) < radius;
            Q _code = static_cast<Q>(posterr + radius);
            outlier[id] = (1 - quantizable) * _s[i1 + 1][i0 + 1];
            bcode[id] = quantizable * _code;
          }
        }
      }
      else
      { // sequential case
#ifdef AVX512
        __m512 vradius, vebx2, vzero;
        if (vector_reg == 512)
        {
          __m512 vradius = _mm512_set1_ps(radius);
          __m512 vebx2 = _mm512_set1_ps(ebs_L4[EBx2_r]);
          __m512 vzero = _mm512_setzero_ps();
        }
#endif
        __m256 vradius8 = _mm256_set1_ps(radius);
        __m256 vebx2_8 = _mm256_set1_ps(ebs_L4[EBx2_r]);
        __m256i mask = _mm256_set1_epi32(0xFFFFFFFF);

        //prequantization
        for (size_t i1 = 0; i1 < blksz; i1++)
        {
          size_t i0 = 0;
          size_t id = _idx0 + (_idx1 + i1) * dims_L16[DIM0];
          size_t blk_end = _idx0 + (_idx1 + i1) * dims_L16[DIM0] + blksz;
          size_t blk_end16 = (blk_end & ~0xF);
          size_t blk_end8 = (blk_end & ~0x7);

#ifdef AVX512
          if (vector_reg == 512)
          {
            for (; id < blk_end16; id += 16, i0 += 16)
            { //AVX-512
              __m512 vdata = _mm512_loadu_ps(&data[id]);
              __m512 s = _mm512_roundscale_ps(_mm512_mul_ps(vdata, vebx2), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
              _mm512_storeu_ps(&_s[i1 + 1][i0 + 1], s);
            }
          }
#endif
          for (; id < blk_end8; id += 8, i0 += 8)
          { //AVX2
            __m256 vdata = _mm256_loadu_ps(&data[id]);
            __m256 s = _mm256_round_ps(_mm256_mul_ps(vdata, vebx2_8), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            _mm256_storeu_ps(&_s[i1 + 1][i0 + 1], s);
          }
          for (; i0 < blksz; i0++)
          { //Sequential Case
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
              continue;
            size_t id = gi0 + gi1 * dims_L16[DIM0];
            _s[i1 + 1][i0 + 1] = round(data[id] * ebs_L4[EBx2_r]);
          }
#ifdef PFETCH
          __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + blksz], 0, 0);
#elif PF2
          __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 2 * blksz], 0, 0);
#elif PF4
          __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 4 * blksz], 0, 0);
#elif PF8
          __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 8 * blksz], 0, 0);
#elif PF16
          __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 16 * blksz], 0, 0);
#endif
        }
        //__m256 vend1    = _mm256_set_ps(_idx1+7,_idx1+6,_idx1+5,_idx1+4,_idx1+3,_idx1+2,_idx1+1,_idx1+0);

        // postquantization
        for (size_t i1 = 0; i1 < blksz; i1++)
        {

          size_t i0 = 0;
          size_t gi1 = _idx1 + i1;
          size_t id = _idx0 + gi1 * dims_L16[DIM0];
          size_t blk_end = _idx0 + ((_idx1 + i1) * dims_L16[DIM0]) + blksz;
          size_t blk_end16 = (blk_end & ~0xF);
          size_t blk_end8 = (blk_end & ~0x7);
          size_t blk_end4 = (blk_end & ~0x3);
          //__m256 vend0    = _mm256_set_ps(_idx0+7,_idx0+6,_idx0+5,_idx0+4,_idx0+3,_idx0+2,_idx0+1,_idx0+0);

#ifdef AVX512
          if (vector_reg == 512)
          {
            for (; id < blk_end16; id += 16, i0 += 16)
            { //AVX-512
              size_t gi1 = _idx1 + i1;
              size_t gi0 = _idx0 + i0 + 15;
              if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
                continue;

              __m512 prevW = _mm512_loadu_ps(&_s[i1 + 1][i0]);
              __m512 prevN = _mm512_loadu_ps(&_s[i1][i0 + 1]);
              __m512 prevNW = _mm512_loadu_ps(&_s[i1][i0]);
              __m512 current = _mm512_loadu_ps(&_s[i1 + 1][i0 + 1]);

              __m512 vpred = _mm512_add_ps(prevW, _mm512_sub_ps(prevN, prevNW));
              __m512 vposterr = _mm512_sub_ps(current, vpred);
              __mmask16 vquantizable = _mm512_cmp_ps_mask(_mm512_abs_ps(vposterr), vradius, 1);
              __mmask16 vnotquant = _mm512_cmp_ps_mask(vradius, _mm512_abs_ps(vposterr), 1);
              __m512i _code = _mm512_cvtps_epi32(_mm512_add_ps(vposterr, vradius));
              __m512 voutlier = _mm512_mask_blend_ps(vquantizable, current, vzero);
              __m512i vbcode = _mm512_mask_blend_epi32(vquantizable, _mm512_setzero_epi32(), _code);

              _mm512_storeu_ps(&outlier[id], voutlier);
              _mm512_mask_storeu_epi32(&bcode[id], vquantizable, vbcode);
              _mm512_mask_storeu_epi32(&bcode[id], vnotquant, _mm512_setzero_epi32());
            }
          }
#endif
          for (; id < blk_end8; id += 8, i0 += 8)
          { //AVX-256
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0 + 7;
            if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
              continue;

            __m256 prevW = _mm256_loadu_ps(&_s[i1 + 1][i0]);
            __m256 prevN = _mm256_loadu_ps(&_s[i1][i0 + 1]);
            __m256 prevNW = _mm256_loadu_ps(&_s[i1][i0]);
            __m256 current = _mm256_loadu_ps(&_s[i1 + 1][i0 + 1]);

            __m256 vpred = _mm256_add_ps(prevW, _mm256_sub_ps(prevN, prevNW));
            __m256 vposterr = _mm256_sub_ps(current, vpred);
            __m256 absposterr = _mm256_sqrt_ps(_mm256_mul_ps(vposterr, vposterr));
            __m256 vquant = _mm256_cmp_ps(absposterr, vradius8, 1);
            __m256i _code = _mm256_cvtps_epi32(_mm256_add_ps(vposterr, vradius8));
            __m256 voutlier = _mm256_andnot_ps(vquant, current);
            __m256i vbcode = _mm256_cvtps_epi32(_mm256_and_ps(vquant, _mm256_cvtepi32_ps(_code)));

            _mm256_storeu_ps(&outlier[id], voutlier);
            _mm256_maskstore_epi32(&bcode[id], mask, vbcode);
          }

          for (; i0 < blksz; i0++)
          {
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
              continue;
            size_t id = gi0 + gi1 * dims_L16[DIM0];
            T pred = _s[i1 + 1][i0] + _s[i1][i0 + 1] - _s[i1][i0];
            T posterr = _s[i1 + 1][i0 + 1] - pred;
            bool quantizable = fabs(posterr) < radius;
            Q _code = static_cast<Q>(posterr + radius);
            outlier[id] = (1 - quantizable) * _s[i1 + 1][i0 + 1];
            bcode[id] = quantizable * _code;
          }
          // vend1 = _mm256_add_ps(vend1,vincr);
        }
      }

    }

    template <typename T, typename Q>
    void c_lorenzo_3d1l(T *data,
                        T *outlier,
                        Q *bcode,
                        size_t const *const dims_L16,
                        double const *const ebs_L4,
                        size_t b0,
                        size_t b1,
                        size_t b2,
                        size_t blksz,
                        int vector_reg,
                        struct SZWorkflow szwf,
                        T pad_constant,
                        int pad_type,
                        T* pad_vals,
                        size_t* pad_idx)
    {
      alignas(32) T _s[blksz + 1][blksz + 1][blksz + 1] {0};

      T* _sptr;
      if (szwf.block_padding or szwf.global_padding)
      {
        T block_pad_value;
        T* block = padding::fill_3d_block<T>(data, dims_L16, blksz, b0, b1, b2);
        _sptr = padding::block_pad<T>(block, dims_L16[nDIM], pad_type, blksz, pad_constant, ebs_L4[EBx2_r], &block_pad_value);
        memcpy(_s, _sptr, sizeof(T) * (blksz + 1) * (blksz + 1) * (blksz + 1));
        if (szwf.block_padding) pad_vals[(*pad_idx)++] = block_pad_value;
        free(block);
      }
      else if (szwf.edge_padding)
      {
        T* edge_pad_values = (T *)malloc(sizeof(T) * dims_L16[nDIM]);
        size_t pad_dims[3] {(size_t)blksz, (size_t)blksz, (size_t)blksz};
        T* block = padding::fill_3d_block<T>(data, dims_L16, blksz, b0, b1, b2);
        _sptr = padding::edge_pad<T>(block, dims_L16[nDIM], pad_type, pad_dims, blksz, pad_constant, ebs_L4[EBx2_r], &edge_pad_values);
        memcpy(_s, _sptr, sizeof(T) * (blksz + 1) * (blksz + 1) * (blksz + 1));
        for (size_t i = 0; i < dims_L16[nDIM]; i++) pad_vals[(*pad_idx)++] = edge_pad_values[i];
        free(block);
      }
      else memset(_s, 0, (blksz + 1) * (blksz + 1) * sizeof(T));
      auto radius = static_cast<Q>(dims_L16[RADIUS]);

      size_t _idx2 = b2 * blksz;
      size_t _idx1 = b1 * blksz;
      size_t _idx0 = b0 * blksz;

      if (_idx2 + blksz < dims_L16[DIM2] and _idx1 + blksz < dims_L16[DIM1] and _idx0 + blksz < dims_L16[DIM0])
      { // vectorizable cases
#ifdef AVX512
        __m512 vradius, vebx2, vzero;
        if (vector_reg == 512)
        {
          __m512 vradius = _mm512_set1_ps(radius);
          __m512 vebx2 = _mm512_set1_ps(ebs_L4[EBx2_r]);
          __m512 vzero = _mm512_setzero_ps();
        }
#endif
        __m256 vradius8 = _mm256_set1_ps(radius);
        __m256 vebx2_8 = _mm256_set1_ps(ebs_L4[EBx2_r]);
        __m256i mask = _mm256_set1_epi32(0xFFFFFFFF);

        // prequantization
        for (size_t i2 = 0; i2 < blksz; i2++)
        {
          for (size_t i1 = 0; i1 < blksz; i1++)
          {
            size_t i0 = 0;
            size_t id = _idx0 + (_idx1 + i1) * dims_L16[DIM0] +
                        (_idx2 + i2) * dims_L16[DIM1] * dims_L16[DIM0];
            size_t blk_end = _idx0 + (_idx1 + i1) * dims_L16[DIM0] +
                             (_idx2 + i2) * dims_L16[DIM1] * dims_L16[DIM0] + blksz;
            size_t blk_end16 = (blk_end & ~0xF);
            size_t blk_end8 = (blk_end & ~0x7);

#ifdef AVX512
            if (vector_reg == 512)
            {
              for (; id < blk_end16; id += 16, i0 += 16)
              { //AVX-512
                __m512 vdata = _mm512_loadu_ps(&data[id]);
                __m512 s = _mm512_roundscale_ps(_mm512_mul_ps(vdata, vebx2), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm512_storeu_ps(&_s[i2 + 1][i1 + 1][i0 + 1], s);
              }
            }
#endif
            for (; id < blk_end8; id += 8, i0 += 8)
            { //AVX2
              __m256 vdata = _mm256_loadu_ps(&data[id]);
              __m256 s = _mm256_round_ps(_mm256_mul_ps(vdata, vebx2_8), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
              _mm256_storeu_ps(&_s[i2 + 1][i1 + 1][i0 + 1], s);
            }
            for (; i0 < blksz; i0++)
            { //Sequential Case
              size_t gi2 = _idx2 + i2;
              size_t gi1 = _idx1 + i1;
              size_t gi0 = _idx0 + i0;
              size_t id = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM1] * dims_L16[DIM0];
              _s[i2 + 1][i1 + 1][i0 + 1] = round(data[id] * ebs_L4[EBx2_r]);
            }
#ifdef PFETCH
            __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + blksz], 0, 0);
#elif PF2
            __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 2 * blksz], 0, 0);
#elif PF4
            __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 4 * blksz], 0, 0);
#elif PF8
            __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 8 * blksz], 0, 0);
#elif PF16
            __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 16 * blksz], 0, 0);
#endif
          }
        }

        // postquantization
        for (size_t i2 = 0; i2 < blksz; i2++)
        {
          for (size_t i1 = 0; i1 < blksz; i1++)
          {
            size_t i0 = 0;
            size_t id = _idx0 + (_idx1 + i1) * dims_L16[DIM0] +
                        (_idx2 + i2) * dims_L16[DIM1] * dims_L16[DIM0];
            size_t blk_end = _idx0 + (_idx1 + i1) * dims_L16[DIM0] +
                             (_idx2 + i2) * dims_L16[DIM1] * dims_L16[DIM0] + blksz;
            size_t blk_end16 = (blk_end & ~0xF);
            size_t blk_end8 = (blk_end & ~0x7);

#ifdef AVX512
            if (vector_reg == 512)
            {
              for (; id < blk_end16; id += 16, i0 += 16)
              { //AVX-512
                __m512 prevXYZ = _mm512_loadu_ps(&_s[i2][i1][i0]);
                __m512 prevXY = _mm512_loadu_ps(&_s[i2 + 1][i1][i0]);
                __m512 prevXZ = _mm512_loadu_ps(&_s[i2][i1 + 1][i0]);
                __m512 prevYZ = _mm512_loadu_ps(&_s[i2][i1][i0 + 1]);
                __m512 prevX = _mm512_loadu_ps(&_s[i2 + 1][i1 + 1][i0]);
                __m512 prevY = _mm512_loadu_ps(&_s[i2 + 1][i1][i0 + 1]);
                __m512 prevZ = _mm512_loadu_ps(&_s[i2][i1 + 1][i0 + 1]);
                __m512 current = _mm512_loadu_ps(&_s[i2 + 1][i1 + 1][i0 + 1]);

                __m512 dist2 = _mm512_add_ps(prevXY, _mm512_add_ps(prevXZ, prevYZ));
                __m512 dist1 = _mm512_add_ps(prevX, _mm512_add_ps(prevY, prevZ));
                __m512 vpred = _mm512_add_ps(_mm512_sub_ps(prevXYZ, dist2), dist1);
                __m512 vposterr = _mm512_sub_ps(current, vpred);
                __mmask16 vquant = _mm512_cmp_ps_mask(_mm512_abs_ps(vposterr), vradius, 1);
                __mmask16 nquant = _mm512_cmp_ps_mask(vradius, _mm512_abs_ps(vposterr), 1);
                __m512i _vcode = _mm512_cvtps_epi32(_mm512_add_ps(vposterr, vradius));
                __m512 voutlier = _mm512_mask_blend_ps(vquant, current, vzero);
                __m512i vbcode = _mm512_mask_blend_epi32(vquant, _mm512_setzero_epi32(), _vcode);

                _mm512_storeu_ps(&outlier[id], voutlier);
                _mm512_mask_storeu_epi32(&bcode[id], vquant, vbcode);
                _mm512_mask_storeu_epi32(&bcode[id], nquant, _mm512_setzero_epi32());
              }
            }
#endif

            for (; id < blk_end8; id += 8, i0 += 8)
            { //AVX2
              __m256 prevXYZ = _mm256_loadu_ps(&_s[i2][i1][i0]);
              __m256 prevXY = _mm256_loadu_ps(&_s[i2 + 1][i1][i0]);
              __m256 prevXZ = _mm256_loadu_ps(&_s[i2][i1 + 1][i0]);
              __m256 prevYZ = _mm256_loadu_ps(&_s[i2][i1][i0 + 1]);
              __m256 prevX = _mm256_loadu_ps(&_s[i2 + 1][i1 + 1][i0]);
              __m256 prevY = _mm256_loadu_ps(&_s[i2 + 1][i1][i0 + 1]);
              __m256 prevZ = _mm256_loadu_ps(&_s[i2][i1 + 1][i0 + 1]);
              __m256 current = _mm256_loadu_ps(&_s[i2 + 1][i1 + 1][i0 + 1]);

              __m256 dist1 = _mm256_add_ps(prevX, _mm256_add_ps(prevY, prevZ));
              __m256 dist2 = _mm256_add_ps(prevXY, _mm256_add_ps(prevXZ, prevYZ));
              __m256 vpred = _mm256_add_ps(_mm256_sub_ps(prevXYZ, dist2), dist1);
              __m256 vposterr = _mm256_sub_ps(current, vpred);
              __m256 absposterr = _mm256_sqrt_ps(_mm256_mul_ps(vposterr, vposterr));
              __m256 vquant = _mm256_cmp_ps(absposterr, vradius8, 1);
              __m256i _code = _mm256_cvtps_epi32(_mm256_add_ps(vposterr, vradius8));
              __m256 voutlier = _mm256_andnot_ps(vquant, current);
              __m256i vbcode = _mm256_cvtps_epi32(_mm256_and_ps(vquant, _mm256_cvtepi32_ps(_code)));

              _mm256_storeu_ps(&outlier[id], voutlier);
              _mm256_maskstore_epi32(&bcode[id], mask, vbcode);
            }
            for (; i0 < blksz; i0++)
            { //sequential case
              size_t gi2 = _idx2 + i2;
              size_t gi1 = _idx1 + i1;
              size_t gi0 = _idx0 + i0;
              size_t id = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM1] * dims_L16[DIM0];
              T pred = _s[i2][i1][i0]                                                              // +, dist=3
                       - _s[i2 + 1][i1][i0] - _s[i2][i1 + 1][i0] - _s[i2][i1][i0 + 1]              // -, dist=2
                       + _s[i2 + 1][i1 + 1][i0] + _s[i2 + 1][i1][i0 + 1] + _s[i2][i1 + 1][i0 + 1]; // +, dist=1
              T posterr = _s[i2 + 1][i1 + 1][i0 + 1] - pred;
              bool quantizable = fabs(posterr) < radius;
              Q _code = static_cast<Q>(posterr + radius);
              outlier[id] = (1 - quantizable) * _s[i2 + 1][i1 + 1][i0 + 1];
              bcode[id] = quantizable * _code;
            }
          }
        }
      }
      else
      {
#ifdef AVX512
        __m512 vradius, vebx2, vzero;
        if (vector_reg == 512)
        {
          __m512 vradius = _mm512_set1_ps(radius);
          __m512 vebx2 = _mm512_set1_ps(ebs_L4[EBx2_r]);
          __m512 vzero = _mm512_setzero_ps();
        }
#endif
        __m256 vradius8 = _mm256_set1_ps(radius);
        __m256 vebx2_8 = _mm256_set1_ps(ebs_L4[EBx2_r]);

        // prequantization
        for (size_t i2 = 0; i2 < blksz; i2++)
        {
          for (size_t i1 = 0; i1 < blksz; i1++)
          {
            size_t i0 = 0;
            size_t id = _idx0 + (_idx1 + i1) * dims_L16[DIM0] +
                        (_idx2 + i2) * dims_L16[DIM1] * dims_L16[DIM0];
            size_t blk_end = _idx0 + (_idx1 + i1) * dims_L16[DIM0] +
                             (_idx2 + i2) * dims_L16[DIM1] * dims_L16[DIM0] + blksz;
            size_t blk_end16 = (blk_end & ~0xF);
            size_t blk_end8 = (blk_end & ~0x7);

#ifdef AVX512
            if (vector_reg == 512)
            {
              for (; id < blk_end16; id += 16, i0 += 16)
              { //AVX-512
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0 + 15;
                if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
                  break;
                __m512 vdata = _mm512_loadu_ps(&data[id]);
                __m512 s = _mm512_roundscale_ps(_mm512_mul_ps(vdata, vebx2), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                _mm512_storeu_ps(&_s[i2 + 1][i1 + 1][i0 + 1], s);
              }
            }
#endif
            for (; id < blk_end8; id += 8, i0 += 8)
            { //AVX2
              size_t gi2 = _idx2 + i2;
              size_t gi1 = _idx1 + i1;
              size_t gi0 = _idx0 + i0 + 7;
              if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
                break;
              __m256 vdata = _mm256_loadu_ps(&data[id]);
              __m256 s = _mm256_round_ps(_mm256_mul_ps(vdata, vebx2_8), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

              _mm256_storeu_ps(&_s[i2 + 1][i1 + 1][i0 + 1], s);
            }
            for (; i0 < blksz; i0++)
            { //Sequential Case
              size_t gi2 = _idx2 + i2;
              size_t gi1 = _idx1 + i1;
              size_t gi0 = _idx0 + i0;
              size_t id = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM1] * dims_L16[DIM0];
              if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
                continue;
              _s[i2 + 1][i1 + 1][i0 + 1] = round(data[id] * ebs_L4[EBx2_r]);
            }
#ifdef PFETCH
            __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + blksz], 0, 0);
#elif PF2
            __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 2 * blksz], 0, 0);
#elif PF4
            __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 4 * blksz], 0, 0);
#elif PF8
            __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 8 * blksz], 0, 0);
#elif PF16
            __builtin_prefetch(&data[(_idx1 + i1) * dims_L16[DIM0] + _idx0 + 16 * blksz], 0, 0);
#endif
          }
        }

        const __m256i mask = _mm256_set1_epi32(0xFFFFFFFF);

        // postquantization
        for (size_t i2 = 0; i2 < blksz; i2++)
        {
          for (size_t i1 = 0; i1 < blksz; i1++)
          {
            size_t i0 = 0;
            size_t id = _idx0 + (_idx1 + i1) * dims_L16[DIM0] +
                        (_idx2 + i2) * dims_L16[DIM1] * dims_L16[DIM0];
            size_t blk_end = _idx0 + (_idx1 + i1) * dims_L16[DIM0] +
                             (_idx2 + i2) * dims_L16[DIM1] * dims_L16[DIM0] + blksz;
            size_t blk_end16 = (blk_end & ~0xF);
            size_t blk_end8 = (blk_end & ~0x7);

#ifdef AVX512
            if (vector_reg == 512)
            {
              for (; id < blk_end16; id += 16, i0 += 16)
              { //AVX-512
                size_t gi2 = _idx2 + i2;
                size_t gi1 = _idx1 + i1;
                size_t gi0 = _idx0 + i0 + 15;
                if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
                  break;
                __m512 prevXYZ = _mm512_loadu_ps(&_s[i2][i1][i0]);
                __m512 prevXY = _mm512_loadu_ps(&_s[i2 + 1][i1][i0]);
                __m512 prevXZ = _mm512_loadu_ps(&_s[i2][i1 + 1][i0]);
                __m512 prevYZ = _mm512_loadu_ps(&_s[i2][i1][i0 + 1]);
                __m512 prevX = _mm512_loadu_ps(&_s[i2 + 1][i1 + 1][i0]);
                __m512 prevY = _mm512_loadu_ps(&_s[i2 + 1][i1][i0 + 1]);
                __m512 prevZ = _mm512_loadu_ps(&_s[i2][i1 + 1][i0 + 1]);
                __m512 current = _mm512_loadu_ps(&_s[i2 + 1][i1 + 1][i0 + 1]);

                __m512 dist2 = _mm512_add_ps(prevXY, _mm512_add_ps(prevXZ, prevYZ));
                __m512 dist1 = _mm512_add_ps(prevX, _mm512_add_ps(prevY, prevZ));
                __m512 vpred = _mm512_add_ps(_mm512_sub_ps(prevXYZ, dist2), dist1);
                __m512 vposterr = _mm512_sub_ps(current, vpred);
                __mmask16 vquant = _mm512_cmp_ps_mask(_mm512_abs_ps(vposterr), vradius, 1);
                __mmask16 nquant = _mm512_cmp_ps_mask(vradius, _mm512_abs_ps(vposterr), 1);
                __m512i _vcode = _mm512_cvtps_epi32(_mm512_add_ps(vposterr, vradius));
                __m512 voutlier = _mm512_mask_blend_ps(vquant, current, vzero);
                __m512i vbcode = _mm512_mask_blend_epi32(vquant, _mm512_setzero_epi32(), _vcode);

                _mm512_storeu_ps(&outlier[id], voutlier);
                _mm512_mask_storeu_epi32(&bcode[id], vquant, vbcode);
                _mm512_mask_storeu_epi32(&bcode[id], nquant, _mm512_setzero_epi32());
              }
            }
#endif

            for (; id < blk_end8; id += 8, i0 += 8)
            { //AVX2
              size_t gi2 = _idx2 + i2;
              size_t gi1 = _idx1 + i1;
              size_t gi0 = _idx0 + i0 + 7;
              if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
                break;
              __m256 prevXYZ = _mm256_loadu_ps(&_s[i2][i1][i0]);
              __m256 prevXY = _mm256_loadu_ps(&_s[i2 + 1][i1][i0]);
              __m256 prevXZ = _mm256_loadu_ps(&_s[i2][i1 + 1][i0]);
              __m256 prevYZ = _mm256_loadu_ps(&_s[i2][i1][i0 + 1]);
              __m256 prevX = _mm256_loadu_ps(&_s[i2 + 1][i1 + 1][i0]);
              __m256 prevY = _mm256_loadu_ps(&_s[i2 + 1][i1][i0 + 1]);
              __m256 prevZ = _mm256_loadu_ps(&_s[i2][i1 + 1][i0 + 1]);
              __m256 current = _mm256_loadu_ps(&_s[i2 + 1][i1 + 1][i0 + 1]);

              __m256 dist1 = _mm256_add_ps(prevX, _mm256_add_ps(prevY, prevZ));
              __m256 dist2 = _mm256_add_ps(prevXY, _mm256_add_ps(prevXZ, prevYZ));
              __m256 vpred = _mm256_add_ps(_mm256_sub_ps(prevXYZ, dist2), dist1);
              __m256 vposterr = _mm256_sub_ps(current, vpred);
              __m256 absposterr = _mm256_sqrt_ps(_mm256_mul_ps(vposterr, vposterr));
              __m256 vquant = _mm256_cmp_ps(absposterr, vradius8, 1);
              __m256i _code = _mm256_cvtps_epi32(_mm256_add_ps(vposterr, vradius8));
              __m256 voutlier = _mm256_andnot_ps(vquant, current);
              __m256i vbcode = _mm256_cvtps_epi32(_mm256_and_ps(vquant, _mm256_cvtepi32_ps(_code)));

              _mm256_storeu_ps(&outlier[id], voutlier);
              _mm256_maskstore_epi32(&bcode[id], mask, vbcode);
            }
            for (; i0 < blksz; i0++)
            { //sequential case
              size_t gi2 = _idx2 + i2;
              size_t gi1 = _idx1 + i1;
              size_t gi0 = _idx0 + i0;
              size_t id = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM1] * dims_L16[DIM0];
              if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
                continue;
              T pred = _s[i2][i1][i0]                                                              // +, dist=3
                       - _s[i2 + 1][i1][i0] - _s[i2][i1 + 1][i0] - _s[i2][i1][i0 + 1]              // -, dist=2
                       + _s[i2 + 1][i1 + 1][i0] + _s[i2 + 1][i1][i0 + 1] + _s[i2][i1 + 1][i0 + 1]; // +, dist=1
              T posterr = _s[i2 + 1][i1 + 1][i0 + 1] - pred;
              bool quantizable = fabs(posterr) < radius;
              Q _code = static_cast<Q>(posterr + radius);
              outlier[id] = (1 - quantizable) * _s[i2 + 1][i1 + 1][i0 + 1];
              bcode[id] = quantizable * _code;
            }
          }
        }
      }

    }

    template <typename T, typename Q>
    void x_lorenzo_1d1l(T *xdata, T *outlier, Q *bcode, size_t const *const dims_L16, double _2EB, size_t b0, int blksz, SZWorkflow szwf, size_t* pad_idx, T* pad_vals)
    {
      auto radius = static_cast<Q>(dims_L16[RADIUS]);
      size_t _idx0 = b0 * blksz;
      float padding = 0;
      if (szwf.block_padding or szwf.edge_padding)
      {
          padding = pad_vals[(*pad_idx)++];
      }
      else {
          padding = pad_vals[0];
      }
      for (int i0 = 0; i0 < blksz; i0++)
      {
        size_t id = _idx0 + i0;
        if (id >= dims_L16[DIM0])
          continue;
        T pred = id < _idx0 + 1 ? padding : xdata[id - 1];
        xdata[id] = bcode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bcode[id] - radius));
      }
      for (int i0 = 0; i0 < blksz; i0++)
      {
        size_t id = _idx0 + i0;
        if (id >= dims_L16[DIM0])
          continue;
        xdata[id] = xdata[id] * _2EB;
      }
    }

    template <typename T, typename Q>
    void x_lorenzo_2d1l(T *xdata, T *outlier, Q *bcode, size_t const *const dims_L16, double _2EB, size_t b0, size_t b1, int blksz, SZWorkflow szwf, size_t* pad_idx, T* pad_vals)
    {
      alignas(32) T _s[blksz + 1][blksz + 1];
      auto radius = static_cast<Q>(dims_L16[RADIUS]);

      if (szwf.block_padding or szwf.global_padding)
      {
          T block_pad_value = 0;
          block_pad_value = (szwf.block_padding) ? pad_vals[(*pad_idx)++] : pad_vals[0];
          T* _sptr = padding::x_block_pad<T>(dims_L16[nDIM], blksz, block_pad_value);
          memcpy(_s, _sptr, sizeof(T) * (blksz + 1) * (blksz + 1));
      }
      else if (szwf.edge_padding)
      {
        T edge_pad_values[2] {pad_vals[(*pad_idx)++], pad_vals[(*pad_idx)++]};
        T* _sptr = padding::x_edge_pad<T>(dims_L16[nDIM], blksz, edge_pad_values);
        memcpy(_s, _sptr, sizeof(T) * (blksz + 1) * (blksz + 1));
      }
      else memset(_s, 0, (blksz + 1) * (blksz + 1) * sizeof(T));

      size_t _idx1 = b1 * blksz;
      size_t _idx0 = b0 * blksz;

      for (int i1 = 0; i1 < blksz; i1++)
      {
        for (int i0 = 0; i0 < blksz; i0++)
        {
          size_t gi1 = _idx1 + i1;
          size_t gi0 = _idx0 + i0;
          if (gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
            continue;
          const size_t id = gi0 + gi1 * dims_L16[DIM0];
          T pred = _s[i1][i0 + 1] + _s[i1 + 1][i0] - _s[i1][i0];
          _s[i1 + 1][i0 + 1] = bcode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bcode[id] - radius));
          xdata[id] = _s[i1 + 1][i0 + 1] * _2EB;
        }
      }

    }

    template <typename T, typename Q>
    void x_lorenzo_3d1l(T *xdata,
                        T *outlier,
                        Q *bcode,
                        size_t const *const dims_L16, //
                        double _2EB,
                        size_t b0,
                        size_t b1,
                        size_t b2,
                        int blksz,
                        SZWorkflow szwf,
                        size_t* pad_idx,
                        T* pad_vals)
    {
      alignas(32) T _s[blksz + 1][blksz + 1][blksz + 1];
      auto radius = static_cast<Q>(dims_L16[RADIUS]);

      if (szwf.block_padding or szwf.global_padding)
      {
          T block_pad_value = 0;
          block_pad_value = (szwf.block_padding) ? pad_vals[(*pad_idx)++] : pad_vals[0];
          T* _sptr = padding::x_block_pad<T>(dims_L16[nDIM], blksz, block_pad_value);
          memcpy(_s, _sptr, sizeof(T) * (blksz + 1) * (blksz + 1) * (blksz + 1));
      }
      else if (szwf.edge_padding)
      {
        T edge_pad_values[3] {pad_vals[(*pad_idx)++], pad_vals[(*pad_idx)++], pad_vals[(*pad_idx)++]};
        T* _sptr = padding::x_edge_pad<T>(dims_L16[nDIM], blksz, edge_pad_values);
        memcpy(_s, _sptr, sizeof(T) * (blksz + 1) * (blksz + 1) * (blksz + 1));
      }
      else memset(_s, 0, (blksz + 1) * (blksz + 1) * (blksz + 1) * sizeof(T));

      size_t _idx2 = b2 * blksz;
      size_t _idx1 = b1 * blksz;
      size_t _idx0 = b0 * blksz;

      for (int i2 = 0; i2 < blksz; i2++)
      {
        for (int i1 = 0; i1 < blksz; i1++)
        {
          for (int i0 = 0; i0 < blksz; i0++)
          {
            size_t gi2 = _idx2 + i2;
            size_t gi1 = _idx1 + i1;
            size_t gi0 = _idx0 + i0;
            if (gi2 >= dims_L16[DIM2] or gi1 >= dims_L16[DIM1] or gi0 >= dims_L16[DIM0])
              continue;
            size_t id = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM1] * dims_L16[DIM0];
            T pred = _s[i2][i1][i0]                                                              // +, dist=3
                     - _s[i2 + 1][i1][i0] - _s[i2][i1 + 1][i0] - _s[i2][i1][i0 + 1]              // -, dist=2
                     + _s[i2 + 1][i1 + 1][i0] + _s[i2 + 1][i1][i0 + 1] + _s[i2][i1 + 1][i0 + 1]; // +, dist=1
            _s[i2 + 1][i1 + 1][i0 + 1] = bcode[id] == 0 ? outlier[id] : static_cast<T>(pred + (bcode[id] - radius));
            xdata[id] = _s[i2 + 1][i1 + 1][i0 + 1] * _2EB;
          }
        }
      }

    }

  } // namespace predictor_quantizer

} // namespace vecsz

#endif
