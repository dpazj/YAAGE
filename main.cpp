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
    tensor input = {-4.0};
    node x(input);

    auto& z = 2 * x + 2 + x;
    auto& q = z.relu() + z * x;
    auto& h = (z * z).relu();
    auto& y = h + q + q * x; 

    graph g(y);
    g.forwards();
    g.backwards();

    std::cout << "y should be -20, y = ";   y.data()->print();
    std::cout << "dy/dx should be 46, dy/dx = "; x.gradient()->print();
}

void test1()
{
    tensor* input = new tensor({-4.0});
    
    node x(input);

    auto& z = 34 - x;
    auto& c = z.pow(2);
    auto& y = c - 34;

    graph g(y);
    g.forwards();
    g.backwards();

    std::cout << "y should be 1410, y = ";   y.data()->print();
    std::cout << "dy/dx should be -76, dy/dx = "; x.gradient()->print();
}

void test2()
{
    tensor input = {24.0f};
    node x(input);
    node y = x.sigmoid();
    graph g(y);
    g.forwards();
    g.backwards();

    y.data()->print();
    x.gradient()->print();
}

class MoonNet : public model
{
    node& create_model()
    {
        auto& input = create_input_node();

        auto& w1 = create_model_param(2,16);
        auto& w2 = create_model_param(16,16);
        auto& w3 = create_model_param(16,1);

        auto& b1 = create_model_param(1,16);
        auto& b2 = create_model_param(1,16);
        auto& b3 = create_model_param(1,1);

        auto& l1 = (input.dot(w1) + b1).relu(); //layer 1
        auto& l2 = (l1.dot(w2) + b2).relu(); //layer 2
        auto& l3 = (l2.dot(w3) + b3).sigmoid(); //layer 3
        
        return l3;
    }
};

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

void do_moon()
{
    std::vector<tensor> X;
    std::vector<tensor> y;

    std::ifstream moon("./datasets/moon_dataset.txt");
    std::string line;
    while (std::getline(moon, line))
    {
        auto vals = split(line, ',');
        tensor tmp = {atof(vals[0].c_str()), atof(vals[1].c_str())};
        tensor tmp1 = {atof(vals[2].c_str())};
        X.push_back(tmp);
        y.push_back(tmp1); 
    }

    double learning_rate = 0.05;
    unsigned int epoch = 25;
    SDG optim(learning_rate);
    MoonNet model;
    model.train(X,y, optim, epoch, loss::binary_cross_entropy);
    model.evaluate(X,y);
}


class MnistNet : public model
{
    node& create_model()
    {
        auto& input = create_input_node();

        auto& w1 = create_model_param(784,128);
        auto& w2 = create_model_param(128,128);
        auto& w3 = create_model_param(128,10);

        auto& b1 = create_model_param(1,128);
        auto& b2 = create_model_param(1,128);
        auto& b3 = create_model_param(1,10);

        //return input.dot(w1).relu().dot(w2).logsoftmax();

        auto& l1 = (input.dot(w1) + b1).relu(); //layer 1
        auto& l2 = (l1.dot(w2) + b2).relu(); //layer 2
        auto& l3 = (l2.dot(w3) + b3).sigmoid(); //layer 3
        
        return l3;
    }
};



void do_mnist()
{
    std::vector<tensor> X;
    std::vector<tensor> y;

    std::ifstream train_x_file("./datasets/mnist/train-images-idx3-ubyte", std::ios::binary);
    std::ifstream train_y_file("./datasets/mnist/train-labels-idx1-ubyte", std::ios::binary);

    std::cout << "loading dataset..." << std::endl;
    std::vector<char> train_x_bytes((std::istreambuf_iterator<char>(train_x_file)), (std::istreambuf_iterator<char>()));
    std::vector<char> train_y_bytes((std::istreambuf_iterator<char>(train_y_file)), (std::istreambuf_iterator<char>()));

    //get rid format bytes
    train_x_bytes.erase(train_x_bytes.begin(), train_x_bytes.begin() + 16); 
    train_y_bytes.erase(train_y_bytes.begin(), train_y_bytes.begin() + 8);

    train_x_file.close();
    train_y_file.close();
    std::cout << "done loading dataset" << std::endl;
    std::cout << "traning samples: " << (train_x_bytes.size()) / 784 << std::endl;

    std::cout << train_x_bytes.size() << std::endl;

    //get labels
    for(size_t i = 0; i < train_y_bytes.size(); i++) //train_y_bytes.size()
    {
        size_t index = (size_t) train_y_bytes[i];
        tensor tmp(1,10);
        tmp.zeros();
        tmp[index] = 1.0f;
        y.push_back(tmp);
    }

    //get inputs
    size_t img_size = 28 * 28;
    for(size_t i = 0; i < train_x_bytes.size(); i+=img_size) //train_x_bytes.size()
    {
        tensor tmp(1,img_size);
        for(size_t j = 0; j < img_size; j++)
        {
            tmp[j] = (double) (unsigned char) train_x_bytes[i + j] / 255; //scale to 0-1;
        }
        X.push_back(tmp);
    }

    double learning_rate = 0.001;
    unsigned int epoch = 10;
    SDG optim(learning_rate);
    MnistNet model;
    
    model.train(X,y, optim, epoch, loss::mean_squared_error);
    

}

int main()
{
    //sanity_test();
    //test1();
    //test2();
    
    //do_moon();

    do_mnist();

    utils::clean_session();

    return 0;

}

