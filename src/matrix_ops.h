#include "matrix.h"

namespace op
{
    Matrix Add(Matrix& a, Matrix& b);
    Matrix Sub(Matrix& a, Matrix& b);
    Matrix Mul(Matrix& a, Matrix& b);
    Matrix Mul(Matrix& a, double b);
    Matrix Dot(Matrix& a, Matrix& b);
    Matrix Sum(Matrix& a);

    Matrix Pow(Matrix& a, double exp);

    Matrix Max(Matrix& a, double val);
    
}