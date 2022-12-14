#include <iostream>
#include <deriv.h>
#include <ndarray.h>
#include <LR.h>

int main() {

    ndarray<float> x({1600, 1});
    ndarray<float> y({1600, 1});
    ndarray<float> theta({400, 1});
    ndarray<float> theta2({400, 1});

    for(int i=0; i<1600; i++) {
        if(i % 2 == 0) {
            x.set({i, 0},i);
            y.set({i, 0} , 1);
        } else {
            x.set({i, 0} , i);
            y.set({i, 0} , 0);
        }
    }

    for (int i = 0; i < 400; i++)
    {
        theta.set({i, 0}, i * 2);
        theta2.set({i, 0}, 1);
    }
    

    LogisticRegression<float> lr(x, y);

    lr.fit(10000, 0.0001);
    std::cout << std::endl;

    std::cout << lr.getWeights() << std::endl;
    std::cout << lr.getBias() << std::endl;

    std::cout <<  "Train Accuracy: "<< lr.accuracy() << std::endl;
    std::cout <<  "Test Accuracy: "<< lr.accuracy(theta, theta2) << std::endl;

    return 0;
}