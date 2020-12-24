#include "src/matrix.h"
#include "src/node.h"
#include "src/graph.h"

int main()
{
    size_t height = 1;
    size_t width = 1;

    Matrix x(height, width);
    Matrix y(height, width);

    for(size_t i=0; i < height;i++)
    {
        for(size_t j=0;j<width;j++)
        {
            x[i][j] = (i * width) + j + 1;
            y[j][i] = (j * height) + i + 1;
        }
    }


    Value a(x);
    Value b(y);

    //create node graph
    Add c(a, b);
    Add d(c, b);
    Pow e(d, 2.0f);
    Add f(e, d);

    Graph graph(a, f);

    auto ans = graph.Forward();
    if(ans != nullptr)
    {
        ans->Print();
    }

    return 0;

}

