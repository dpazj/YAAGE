#include "src/tensor.h"
#include "src/node.h"
#include "src/graph.h"

#include <iostream>

int main()
{
    



    Value A = {{1, 0}, 
               {0, -1}};

    Value b = {{1},
               {1}};

    Value x= {{1},
              {2}};

    Dot a1(A, x);
    Add z(a1,b);



    Graph graph(a1, z);

    auto ans = graph.Forward();
    if(ans != nullptr)
    {
        std::cout << std::endl;
        ans->Print();
    }

    return 0;

}

