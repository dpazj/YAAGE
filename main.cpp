#include "src/matrix.h"
#include "src/node.h"

int main()
{
    size_t height = 5;
    size_t width = 10;

    Matrix x(height, width);
    Matrix y(height, width);

    for(size_t i=0; i < height;i++)
    {
        for(size_t j=0;j<width;j++)
        {
            x[i][j] = i + j;
            y[i][j] = i + j;
        }
    }

    
    
}

