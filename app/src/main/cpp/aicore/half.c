#include <stdint.h>
#include <math.h>
#include "half.h"

#ifdef OPENCL
cl_half to_half(float f)
{
	const struct {
		unsigned int bit_size;     
		unsigned int num_frac_bits;
		unsigned int num_exp_bits; 
		unsigned int sign_bit;    
		unsigned int sign_mask;   
		unsigned int frac_mask;   
		unsigned int exp_mask;    
		unsigned int e_max;     
		int          e_min;         
		unsigned int max_normal;    
		unsigned int min_normal;    
		unsigned int bias_diff;     
		unsigned int frac_bits_diff;
	} float16_params = {
		16,                                                
		10,                                                
		5,                                                 
		15,                                                
		1 << 15,                                          
		(1 << 10) - 1,                                     
		((1 << 5) - 1) << 10,                             
		(1 << (5 - 1)) - 1,                                
		-((1 << (5 - 1)) - 1) + 1,                         
		((((1 << (5 - 1)) - 1) + 127) << 23) | 0x7FE000,  
		((-((1 << (5 - 1)) - 1) + 1) + 127) << 23,         
		((unsigned int)(((1 << (5 - 1)) - 1) - 127) << 23),
		23 - 10                                           
	};
	
	const struct {
		unsigned int abs_value_mask;   
		unsigned int sign_bit_mask;    
		unsigned int e_max;            
		unsigned int num_mantissa_bits;
		unsigned int mantissa_mask;    
	} float32_params = {
		0x7FFFFFFF,
		0x80000000,
		127,       
		23,        
		0x007FFFFF
	};
	
	const union {
		float f;
		unsigned int bits;
	} value = {f};
	
	const unsigned int f_abs_bits = value.bits & float32_params.abs_value_mask;
    const int is_neg = value.bits & float32_params.sign_bit_mask;
    const unsigned int sign = (value.bits & float32_params.sign_bit_mask) >> (float16_params.num_frac_bits +
		float16_params.num_exp_bits + 1);
    cl_half h = 0;

	if (isnan(value.f)) {
		h = float16_params.exp_mask | float16_params.frac_mask;
    } else if (isinf(value.f)) {
		h = is_neg ? float16_params.sign_mask | float16_params.exp_mask : float16_params.exp_mask;
    } else if (f_abs_bits > float16_params.max_normal) {
		h = sign | (((1 << float16_params.num_exp_bits) - 1) << float16_params.num_frac_bits) | float16_params.frac_mask;
    } else if (f_abs_bits < float16_params.min_normal) {
		const unsigned int frac_bits = (f_abs_bits & float32_params.mantissa_mask) | (1 << float32_params.num_mantissa_bits);
		const int nshift = float16_params.e_min + float32_params.e_max - (f_abs_bits >> float32_params.num_mantissa_bits);
		const unsigned int shifted_bits = nshift < 24 ? frac_bits >> nshift : 0;
		h = sign | (shifted_bits >> float16_params.frac_bits_diff);
    } else {
		h = sign | ((f_abs_bits + float16_params.bias_diff) >> float16_params.frac_bits_diff);
    }
	
    return h;
}

float to_float(cl_half h)
{
	const struct {
		uint16_t sign_mask;          
		uint16_t exp_mask;
		int      exp_bias;
		int      exp_offset;
		uint16_t biased_exp_max;
		uint16_t frac_mask;   
		float    smallest_subnormal_as_float;
	} float16_params = {
		0x8000,
		0x7C00,
		15,
		10,
		(1 << 5) - 1,
		0x03FF,
		5.96046448e-8f
	};
	
	const struct {
        int sign_offset;
        int exp_bias;
        int exp_offset;
    } float32_params = {
		31,
		127,
		23
	};
	
	const int is_pos = (h & float16_params.sign_mask) == 0;
    const uint32_t biased_exponent = (h & float16_params.exp_mask) >> float16_params.exp_offset;
    const uint32_t frac = (h & float16_params.frac_mask);
    const int is_inf = biased_exponent == float16_params.biased_exp_max && (frac == 0);

    if (is_inf) {
        return is_pos ? HUGE_VALF : -HUGE_VALF;
    }

    const int is_nan = biased_exponent == float16_params.biased_exp_max && (frac != 0);
    if (is_nan) {
        return NAN;
    }

    const int is_subnormal = biased_exponent == 0;
    if (is_subnormal) {
        return ((float)(frac)) * float16_params.smallest_subnormal_as_float * (is_pos ? 1.f : -1.f);
    }

    const int unbiased_exp = (int)(biased_exponent) - float16_params.exp_bias;
    const uint32_t biased_f32_exponent = (uint32_t)(unbiased_exp + float32_params.exp_bias);

    union {
        cl_float f;
        uint32_t ui;
    } res = {0};

    res.ui = (is_pos ? 0 : 1 << float32_params.sign_offset)
             | (biased_f32_exponent << float32_params.exp_offset)
             | (frac << (float32_params.exp_offset - float16_params.exp_offset));

    return res.f;
}
#endif