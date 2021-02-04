#include <iostream>
#include <iomanip>

#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

#include "cozygrad/cozygrad.h"

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

void unbroadcast_test()
{



    tensor<double> x = {1,2,3,4,5,6,7,8};
    tensor<double> y = {1,2};
    x.reshape({4,2,1});

    node<double> a(x);
    node<double> b(y);


    auto d = (a + b).sum(1);
    auto e = d.sum();

    graph<double> g(e);
    g.forwards();
    g.backwards();

    std::cout << a.gradient() << std::endl;
    std::cout << b.gradient() << std::endl;
    std::cout << d.data() << std::endl;
    std::cout << e.data() << std::endl;

    utils::clean_session<double>();

}

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
    std::cout << a + b << std::endl;
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


void broadcasting_test5()
{
    tensor<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
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
    std::cout << 5.0 / a << std::endl;
    std::cout << a / 5.0 << std::endl;
    std::cout << a * b << std::endl;
    
}

void sum_test1()
{
    tensor<double> a = {1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8};
    
    a.reshape({4,2,3});
    std::cout << a << std::endl;
    // std::cout << op::sum(a,0) << std::endl << std::endl;
    // std::cout << op::sum(a,1) << std::endl << std::endl;
    // std::cout << op::sum(a,2) << std::endl << std::endl;

    std::cout << op::sum(op::sum(a,0),2) << std::endl << std::endl;
    std::cout << op::sum(a, {0,2}) << std::endl << std::endl;
}

void sum_test2()
{
    tensor<double> a = {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10};
    a.reshape({5,2,2,1,5});
    std::cout << a << std::endl;
    std::cout << std::endl;
    std::cout << op::sum(a,0) << std::endl;
    std::cout << op::sum(a,1) << std::endl;
    std::cout << op::sum(a,2) << std::endl;
    std::cout << op::sum(a,3) << std::endl;
    std::cout << op::sum(a,{4}) << std::endl;
}

void equals_test()
{
    tensor<double> a = {1,2,3,4,5};
    tensor<double> b = {1,2,3,4,5};
    tensor<double> c = {1,2,5,4,5};

    std::cout << (a == b) << std::endl;
    std::cout << (a == c) << std::endl;
    b.reshape({1,5});
    std::cout << (a == b) << std::endl;
}


void dot_test1()
{
    tensor<double> a = {1,2,3,4,5,6,7,8,9,10,11,12};
    tensor<double> b = {1,2,3,4,5,6,7,8,9,10,11,12};

    a.reshape({3,4});
    b.reshape({4,3});

    std::cout << a << std::endl;
    std::cout << b << std::endl;

    std::cout << op::dot(a,b) << std::endl;

}

void dot_test2()
{
    tensor<double> a = {1,2,3,4,5,6,7,8,9};
    tensor<double> b = {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2};

    a.reshape({3,3});
    b.reshape({2,3,3});

    std::cout << a << std::endl;
    std::cout << b << std::endl;

    std::cout << op::dot(a,b) << std::endl;

}

void transpose_test1()
{
    tensor<double> a = {1,2,3,4,5,6,7,8,9};
    tensor<double> b = {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2};

    a.reshape({3,3});
    b.reshape({2,3,3});

    std::cout << a << std::endl;
    std::cout << b << std::endl;

    std::cout << op::transpose(a) << std::endl;
    std::cout << op::transpose(b) << std::endl;
}

void tensor_test()
{
    // broadcasting_test1();
    // broadcasting_test2();
    // broadcasting_test3();
    // broadcasting_test5();
    // broacasting_test6();

//     sum_test1();
    //sum_test2();

    // equals_test();
    // dot_test1();
    // dot_test2();

    transpose_test1();
}


class IA
{
    public:
        virtual ~IA(){std::cout << "IA destructor" << std::endl;}
};

class A : public IA
{
    public:
        ~A()
        {
            std::cout << "A destructor" << std::endl;
        }
};

int main()
{
    //tensor_test();
    //sanity_test();

    //unbroadcast_test();


    A* a = new A();

    IA* b = a;

    delete b;

    //test1();
    //test2();
    //test3();
 
    //utils::clean_session();

    return 0;

}

