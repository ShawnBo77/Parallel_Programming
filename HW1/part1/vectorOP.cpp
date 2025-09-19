#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_int zero_int = _pp_vset_int(0);
  __pp_vec_int one_int = _pp_vset_int(1);
  __pp_vec_float limit = _pp_vset_float(9.999999f);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // 計算遮罩
    int remaining = N - i;
    int width = (remaining < VECTOR_WIDTH) ? remaining : VECTOR_WIDTH;
    __pp_mask valid_mask = _pp_init_ones(width);

    __pp_vec_float x;
    _pp_vload_float(x, values + i, valid_mask);
    __pp_vec_int y;
    _pp_vload_int(y, exponents + i, valid_mask);
    __pp_vec_float result;
    _pp_vset_float(result, 1.f, valid_mask); 

    // 次方運算
    // active_mask 是 valid_mask 和 (y > 0) 的交集
    __pp_mask gt_zero_mask;
    _pp_vgt_int(gt_zero_mask, y, zero_int, valid_mask); 
    __pp_mask active_mask = _pp_mask_and(gt_zero_mask, valid_mask);

    while (_pp_cntbits(active_mask) > 0)
    {
      _pp_vmult_float(result, result, x, active_mask);
      _pp_vsub_int(y, y, one_int, active_mask);
      
      // 更新 active_mask
      _pp_vgt_int(gt_zero_mask, y, zero_int, active_mask);
      active_mask = _pp_mask_and(gt_zero_mask, active_mask);
    }

    // 處理超過 limit 的值
    __pp_mask clamp_mask;
    _pp_vgt_float(clamp_mask, result, limit, valid_mask); 
    _pp_vmove_float(result, limit, clamp_mask);

    _pp_vstore_float(output + i, result, valid_mask);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  // for (int i = 0; i < N; i += VECTOR_WIDTH)
  // {
  // }
  
  return 0.0;
}