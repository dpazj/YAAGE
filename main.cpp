#include "src/matrix.h"
#include "src/matrix_ops.h"
#include "src/node.h"
#include "src/graph.h"

#include <iostream>
#include <iomanip>



void SanityTest()
{
    Value x({-4.0});
    Value two({2.0});
    //z
    Mul x_mul_two(x,two);
    Add x_add_two(x,two);
    Add z(x_mul_two, x_add_two); 

    //q
    ReLU z_relu(z);
    Mul z_mul_x(z,x);
    Add q(z_mul_x, z_relu);

    //h
    Mul z_mul_z(z,z);
    ReLU h(z_mul_z);

    //y
    Mul q_mul_x(q,x);
    Add q_add_qmulx(q_mul_x,q);
    Add y(q_add_qmulx, h);
  
    Graph graph(x,y);
    graph.Forward();
    auto* answer = graph.Forward();
    std::cout << "y should be -20, y = ";   answer->Print();
    auto* gradient_input = graph.Backward();
    std::cout << "dy/dx should be 46, dy/dx = "; gradient_input->Print();
    
}

void OtherTest()
{
    Value x({-4.0});

    auto y = x + x + x;

    Graph g(x,y);
    g.Forward();
    auto* answer = g.Forward();
    std::cout << "y should be 12, y = ";   answer->Print();
}



int main()
{
    OtherTest();
    //SanityTest();
    
    return 0;

}

