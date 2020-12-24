#include "src/tensor.h"
#include "src/node.h"
#include "src/graph.h"

#include <iostream>

int main()
{
    

    Tensor A = {{1, 0}, 
                {0, -1}};

    Tensor b = {{1},
                {1}};

    Tensor x = {{1},
                {2}};


    Value nodeA(A);
    Value nodeb(b);
    Value nodex(x);

    MatMul a1(nodeA, nodex);
    Add z(a1,nodeb);



    Graph graph(a1, z);

    auto ans = graph.Forward();
    if(ans != nullptr)
    {
        std::cout << std::endl;
        ans->Print();
    }

    return 0;

}

