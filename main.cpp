#include "src/tensor.h"
#include "src/tensor_ops.h"
#include "src/node.h"
#include "src/graph.h"

#include <iostream>
#include <iomanip>

int main()
{
    
    std::cout << std::fixed << std::setprecision(4) << sizeof(double) << std::endl;
    Value x({4.0});
    
    Pow y(x,6.0);
    Pow y1(y, 2.0);
    Pow y2(y1, 2.0);

    Graph graph(x,y2);

    auto* answer = graph.Forward();
    
    std::cout << "y = ";   answer->Print();
    
    auto* gradient_input = graph.Backward();
    std::cout << "dy/dx = "; gradient_input->Print();
    


    return 0;

}

