#include "matrix.h"

namespace op
{
    Matrix Add(const Matrix& a, const Matrix& b);
    Matrix Sub(const Matrix& a, const Matrix& b);
    Matrix Mul(const Matrix& a, const Matrix& b);
    Matrix Mul(const Matrix& a, double b);
    Matrix Dot(const Matrix& a, const Matrix& b);

    Matrix Pow(const Matrix& a, double exp);
    Matrix Max(const Matrix& a, double val);
    Matrix Transpose(const Matrix& a);
    
}