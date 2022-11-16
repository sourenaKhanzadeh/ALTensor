#include <iostream>
#include <deriv.h>
#include <ndarray.h>
#include <LR.h>

int main() {

    ndarray<float> x({2, 100});
    ndarray<float> y({2, 100});

    x.random();
    y.random();

    LinearRegression<float> lr(x, y);

    lr.fit(1000, 0.01);

    std::cout << lr.getWeights() << std::endl;
    std::cout << lr.getBias() << std::endl;
    


    return 0;
}