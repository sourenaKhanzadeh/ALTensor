#include <iostream>
#include <deriv.h>
#include <ndarray.h>
#include <LR.h>

int main() {

    ndarray<float> x({10, 1});
    ndarray<float> y({10, 1});

    for(int i=0; i<10; i++) {
        x.set({i, 0}, i);
        y.set({i, 0}, i*2);
    }

    LinearRegression<float> lr(x, y);

    lr.fit(10000, 0.0001);

    std::cout << lr.getWeights() << std::endl;
    std::cout << lr.getBias() << std::endl;

    std::cout << lr.predict(x).flatten() << std::endl;
    std::cout << y.flatten() << std::endl;

    std::cout <<  "MSE: "<< lr.MSE() << std::endl;
    


    return 0;
}