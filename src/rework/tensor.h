#include <vector>

class tensor
{
    public:
        tensor(double val);


        tensor operator+(tensor& rhs);
        tensor operator*(tensor& rhs);

    private:

        double m_data;
        double m_gradient;

        tensor* m_in;
        std::vector<tensor*> m_out;


}