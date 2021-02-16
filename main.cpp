#include <iostream>
#include <iomanip>

#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

#include "cozygrad/cozygrad.h"

#include "examples/moonnet.hpp"
#include "examples/mnist.hpp"

using namespace czy;

void sanity_test()
{
    tensor<double> input = {-4.0};
    node<double> x(input);

    auto& z = 2.0 * x + 2.0 + x;
    auto& q = z.relu() + z * x;
    auto& h = (z * z).relu();
    auto& y = h + q + q * x; 

    graph<double> g(y);
    g.forwards();
    g.backwards();

    std::cout << "y should be -20, y = " << y.data() << std::endl;
    std::cout << "dy/dx should be 46, dy/dx = " << x.gradient() << std::endl;

    utils::clean_session<double>();
}

void test_loss_fn()
{

    tensor<double> a = { 1 , 0 , 1 , 0, 0, 1, 1, 0};
    tensor<double> b = { 0.997, 0.997 , 0.997, 0.997 ,0.997 ,0.997 , 0.997 ,0.997 };
    a.reshape({4,2});
    b.reshape({4,2});

    node<double> y(a);
    node<double> yhat(b);

    auto loss = loss::mean_squared_error(y, yhat);

    graph<double> g(loss);

    g.forwards();
    g.backwards();

    std::cout << yhat.gradient() << std::endl;
    std::cout << loss.data() << std::endl;


}

int main()
{

    utils::set_random_seed(1337);
    
    std::cout << std::setprecision(5) << std::endl;
    //test();
    do_mnist();
    //do_moon();
   
    return 0;

}

