#include <iostream>
#include <iomanip>

#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

#include "cozygrad/cozygrad.h"

using namespace czy;

// void sanity_test()
// {
//     tensor input = {-4.0};
//     node x(input);

//     auto& z = 2 * x + 2 + x;
//     auto& q = z.relu() + z * x;
//     auto& h = (z * z).relu();
//     auto& y = h + q + q * x; 

//     graph g(y);
//     g.forwards();
//     g.backwards();

//     std::cout << "y should be -20, y = ";   y.data()->print();
//     std::cout << "dy/dx should be 46, dy/dx = "; x.gradient()->print();
// }

// void test1()
// {
//     tensor* input = new tensor({-4.0});
    
//     node x(input);

//     auto& z = 34 - x;
//     auto& c = z.pow(2);
//     auto& y = c - 34;

//     graph g(y);
//     g.forwards();
//     g.backwards();

//     std::cout << "y should be 1410, y = ";   y.data()->print();
//     std::cout << "dy/dx should be -76, dy/dx = "; x.gradient()->print();
// }

// void test2()
// {
//     tensor input = {-24.0f};
//     node x(input);
//     node y = x.sigmoid();
//     graph g(y);
//     g.forwards();
//     g.backwards();

//     y.data()->print();
//     x.gradient()->print();
// }

// void test3()
// {
//     tensor a = {0,0,0,0,0,1,0,0,0,0};
//     //tensor b = {1, 1, -0.000119375, -4.64622e-05, 0.000252165, 0.00224447, 3.8342e-05, 0.000133732, 9.79701e-05 ,3.23683e-05}; 
//     tensor b = {0, 1, -5.68027e-05, 0.000103058, 0.000231966, 7.61339e-05, 0.00019536, -0.000125528, 0.000223787, 9.07328e-05};
//     //tensor b = {1,0,0,0,0,0,0,0,0,0};

//     node y(a);
//     node yh(b);

//     auto z = loss::mean_squared_error(y, yh);

//     graph g(z);
//     g.forwards();
//     g.backwards();

//     z.data()->print();
//     yh.gradient()->print();
// }



void broadcasting_test1()
{

    tensor<double> a = {1, 2, 3};
    tensor<double> b = {2, 2, 2};
    std::cout << a + b << std::endl << std::endl;
}

void broadcasting_test2()
{
    tensor<double> a = {1, 2, 3, 4};
    tensor<double> b = {2.0};
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << a + b << std::endl << std::endl;
}

void broadcasting_test3()
{
    tensor<double> a = {{1, 2, 3}, {4,5,6}, {7,8,9}, {10,11,12}};
    tensor<double> b = {1,0,1};
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << a + b << std::endl;
    std::cout << std::endl;
}

void broadcasting_test4()
{
    tensor<double> a = {{1, 2, 3}, {4,5,6}, {7,8,9}, {10,11,12}};
    tensor<double> b = {1};
    std::cout << a + b << std::endl << std::endl;
}

void broadcasting_test5()
{
    std::vector<double> w_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<char> w_buf(w_data.size() * sizeof(double));
    std::memcpy(w_buf.data(), w_data.data(), w_data.size() * sizeof(double));
    tensor<double> a(w_buf, {16});
    a.reshape({4,2,1,2});
    tensor<double> b = {2,2};
    b.reshape({1,1,1,1,1,2});

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << a + b << std::endl << std::endl;
}

void broacasting_test6()
{
    tensor<double> a = {12,24,36};
    tensor<double> b = {45,55};
    a.reshape({3,1});
    std::cout << a << std::endl << b << std::endl;
    std::cout << a + b << std::endl;
    
}


void tensor_test()
{

    broadcasting_test1();
    broadcasting_test2();
    broadcasting_test3();
    broadcasting_test4();
    broadcasting_test5();
    broacasting_test6();

    // std::vector<double> w_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // std::vector<char> w_buf(w_data.size() * sizeof(double));
    // std::memcpy(w_buf.data(), w_data.data(), w_data.size() * sizeof(double));

    // tensor<double> w(w_buf, {16});
    // std::cout << w << std::endl;
    // w.reshape({4,2,1,2});
    // auto t = w.slice(1,4);
    // std::cout << w << std::endl;
    // std::cout << t << std::endl;

}


int main()
{

    tensor_test();

    //sanity_test();
    //test1();
    //test2();
    //test3();
 
    //utils::clean_session();

    return 0;

}

