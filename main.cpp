// #include "src/matrix.h"
// #include "src/matrix_ops.h"
// #include "src/node.h"
// #include "src/graph.h"

#include <iostream>
#include <iomanip>

#include "cozygrad/cozygrad.h"


void sanity_test()
{
    tensor input = {-4.0};
    tensor input1 = {2.0};
    
    node x(input);
    node two(input1);

    node z = two * x + two + x;
    node q = z.relu() + z * x;
    node h = (z * z).relu();
    node y = h + q + q * x; 

    graph g(x,y);
    g.forwards();
    g.backwards();

    std::cout << "y should be -20, y = ";   y.data()->print();
    std::cout << "dy/dx should be 46, dy/dx = "; x.gradient()->print();
}

void test1()
{
    tensor input = {-4.0};
    tensor input1 = {34.0};

    node x(input);
    node x1(input1);

    auto z = x1 - x;
    auto c = z.pow(2);
    auto y = c - x1;

    graph g(x,y);
    g.forwards();
    g.backwards();

    std::cout << "y should be 1410, y = ";   y.data()->print();
    std::cout << "dy/dx should be -76, dy/dx = "; x.gradient()->print();


}


int main()
{
  
    sanity_test();
    test1();
    
    return 0;

}

