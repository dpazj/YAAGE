#include "src/tensor.h"
#include "src/tensor_ops.h"
#include "src/node.h"
#include "src/graph.h"

#include <iostream>

int main()
{
      
    Value x({4.0});

    Add y(x,x);
    Add z(y,x);

    Graph graph(x,z);

    auto* answer = graph.Forward();
    if(answer != nullptr)
    {
        answer->Print();
    }
    
    auto* gradient_input = graph.Backward();
    if(gradient_input != nullptr)
    {
        gradient_input->Print();
    }
                

    return 0;

}

