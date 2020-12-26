#include "tensor.h"

namespace op
{
    Tensor Add(Tensor& a, Tensor& b);
    Tensor Sub(Tensor& a, Tensor& b);
    Tensor Mul(Tensor& a, Tensor& b);
    Tensor Dot(Tensor& a, Tensor& b);
    Tensor Sum(Tensor& a);

    Tensor Pow(Tensor& a, double exp);
    
}