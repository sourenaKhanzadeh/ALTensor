#include <iostream>
#include <deriv.h>
#include <ndarray.h>
#include <LR.h>

int main() {

    ndarray<float> x({10, 1});
    ndarray<float> y({10, 1});
    ndarray<float> theta({200, 1});

    for(int i=0; i<10; i++) {
        if(i % 2 == 0) {
            x.set({i, 0},i);
            y.set({i, 0} , 1);
        } else {
            x.set({i, 0} , i);
            y.set({i, 0} , 0);
        }
    }

    for (int i = 0; i < 200; i++)
    {
        theta.set({i, 0}, i * 2);
    }
    

    LogisticRegression<float> lr(x, y);

    lr.fit(10000, 0.0001);
    std::cout << std::endl;

    std::cout << lr.getWeights() << std::endl;
    std::cout << lr.getBias() << std::endl;

    std::cout << lr.predict(x).flatten() << std::endl;
    std::cout << y.flatten() << std::endl;

    std::cout <<  "Accuracy: "<< lr.accuracy() << std::endl;
    
    std::cout << "predict(10): " << lr.predict(theta) << std::endl;

    return 0;
}