#include "../cozygrad/cozygrad.h"
#include <vector>
#include <fstream>
#include <sstream>

using namespace czy;

template <typename T>
class MoonNet : public model<T>
{
    node<T>& create_model()
    {
        auto& input = this->create_input_node();

        auto& w1 = this->create_model_param({2,16});
        auto& w2 = this->create_model_param({16,16});
        auto& w3 = this->create_model_param({16,1});

        auto& b1 = this->create_model_param({1,16});
        auto& b2 = this->create_model_param({1,16});
        auto& b3 = this->create_model_param({1,1});

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
    std::vector<double> vec_X;
    std::vector<double> vec_y;

    std::ifstream moon("./datasets/moon_dataset.txt");
    std::string line;

    while (std::getline(moon, line))
    {
        auto vals = split(line, ',');
    
        vec_X.push_back( atof(vals[0].c_str()) );
        vec_X.push_back( atof(vals[1].c_str()) );
        vec_y.push_back( atof(vals[2].c_str()) );
    }


    tensor<double> X(vec_X,{vec_X.size()/2, 2});
    tensor<double> y(vec_y,{vec_y.size(), 1});

    size_t batch_size = 128;
    double learning_rate = 0.5;
    unsigned int epoch = 100;
    SDG<double> optim(learning_rate);
    MoonNet<double> model;
    model.train(X,y, optim, batch_size, epoch, loss::binary_cross_entropy<double>);
    //model.evaluate(X,y);
}
