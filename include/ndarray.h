#ifndef NDARRAY_H
#define NDARRAY_H

#include <vector>
#include <sstream>
#include <random>
#include <iostream>

template <typename T>
class NDArray {
    public:
        NDArray() = default;
        NDArray(std::vector<int> shape);
        NDArray(std::vector<int> shape, std::vector<T> data);
        ~NDArray();
        const T& operator[](const std::vector<int> index) const;
        NDArray<T> operator[](int index) const;
        T operator[](int index);
        // print the array
        friend std::ostream& operator<<(std::ostream& os, const NDArray<T>& arr) {
            os << arr.str(0);
            return os;
        }

        // equals
        NDArray<T>& operator=(const NDArray<T>& other);

        // set a value
        void set(const std::vector<int> index, T value);

        // add two arrays
        NDArray<T> operator+(const NDArray<T>& arr);

        // add a scalar
        NDArray<T> operator+(const T scalar);

        // subtract a scaler
        NDArray<T> operator-(const T scalar);

        // subtract two arrays
        NDArray<T> operator-(const NDArray<T>& arr);

        NDArray<T> operator-= (const NDArray<T>& arr);

        NDArray<T> operator-= (const T scalar);

        bool operator== (const NDArray<T>& arr);

        // multiply two arrays
        NDArray<T> operator*(const NDArray<T>& arr);

        // multiply scalar
        NDArray<T> operator*(T scalar);

        // tensor product of two arrays
        NDArray<T> matMult(const NDArray<T>& arr);

        // divide two arrays
        NDArray<T> operator/(const NDArray<T>& arr);

        // divide a scalar to an array
        NDArray<T> operator/(T scalar);

        // tensor product of two arrays
        NDArray<T> tensProd(const NDArray<T>& arr);

        // expand the array by one dimension
        NDArray<T> expandDims(int axis);

        int size();
        int size(int dim);
        int rank();
        std::vector<int> shape();
        std::vector<int> strides();
        void reshape(std::vector<int> shape);
        void resize(std::vector<int> shape);
        void resize(int size);
        void resize(int size, int dim);
        void resize(int size, int dim, T value);
        void resize(int size, int dim, T value, bool copy);
        void resize(int size, int dim, bool copy);
        void resize(int size, bool copy);
        void resize(std::vector<int> shape, T value);
        void resize(std::vector<int> shape, T value, bool copy);
        void resize(std::vector<int> shape, bool copy);
        void resize(bool copy);
        void fill(T value);
        void fill(T value, bool copy);
        void fill(bool copy);
        void copy(NDArray<T> &other);

        // return a string representation of the array
        std::string str(int index) const;
        void random();
        void random(T min, T max);
        NDArray<T> transpose();
        NDArray<T> transpose(int dim1, int dim2);

        NDArray<T> flatten();
        std::vector<T> toVector();
        NDArray<T> round();
        NDArray<T> abs();
        NDArray<T> exp();
        NDArray<T> pow(int power);
        NDArray<T> sum(int axis);
        NDArray<T> inv();
        NDArray<T> eq(NDArray<T> &y);

        T dot(NDArray<T> &other);
        T sum();

    private:
        std::vector<T> data;
        std::vector<int> shape_;
        std::vector<int> strides_;
        int size_;
        int rank_;
};

// implementation
template <typename T>
NDArray<T>::NDArray(std::vector<int> shape) {
    shape_ = shape;
    rank_ = shape.size();
    strides_.resize(rank_);
    strides_[rank_ - 1] = 1;
    size_ = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
        size_ *= shape_[i];
    }
    size_ *= shape_[0];
    data.resize(size_);
}

template <typename T>
NDArray<T>::NDArray(std::vector<int> shape, std::vector<T> data) {
    shape_ = shape;
    rank_ = shape.size();
    strides_.resize(rank_);
    strides_[rank_ - 1] = 1;
    size_ = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
        size_ *= shape_[i];
    }
    size_ *= shape_[0];
    this->data = data;

}
    

template <typename T>
NDArray<T>::~NDArray() {
    data.clear();
    shape_.clear();
    strides_.clear();
}


template <typename T>
const T& NDArray<T>::operator[](const std::vector<int> index) const{
    int offset = 0;
    for (int i = 0; i < rank_; i++) {
        offset += index[i] * strides_[i];
    }
    return data[offset];
}

template <typename T>
NDArray<T> NDArray<T>::operator[](int index) const{
    std::vector<int> new_shape(shape_.begin() + 1, shape_.end());
    std::vector<T> new_data(data.begin() + index * strides_[0], data.begin() + (index + 1) * strides_[0]);
    return NDArray<T>(new_shape, new_data);
}

template <typename T>
T NDArray<T>::operator[](int index) {
    // return the element at the given index
    return data[index];
}

template <typename T>
NDArray<T>& NDArray<T>::operator=(const NDArray<T>& other) {
    this->data = other.data;
    this->shape_ = other.shape_;
    this->strides_ = other.strides_;
    this->size_ = other.size_;
    this->rank_ = other.rank_;
    return *this;
}

template <typename T>
bool NDArray<T>::operator==(const NDArray<T>& arr) {
    if (this->shape_ != arr.shape_) {
        return false;
    }
    for (int i = 0; i < this->size_; i++) {
        if (this->data[i] != arr.data[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
NDArray<T> NDArray<T>::operator+(const NDArray<T>& arr) {
    // check if the shapes are the same
    if (shape_ != arr.shape_) {
        throw std::invalid_argument("Shapes are not the same");
    }
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] + arr.data[i];
    }
    return NDArray<T>(shape_, new_data);
}

template <typename T>
NDArray<T> NDArray<T>::operator+(const T scalar) {
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] + scalar;
    }
    return NDArray<T>(shape_, new_data);
}

template <typename T>
NDArray<T> NDArray<T>::operator-(const NDArray<T>& arr) {
    // check if the shapes are the same
    if (shape_ != arr.shape_) {
        throw std::invalid_argument("Shapes are not the same");
    }
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] - arr.data[i];
    }
    return NDArray<T>(shape_, new_data);
}

template <typename T>
NDArray<T> NDArray<T>::operator-(const T scalar) {
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] - scalar;
    }
    return NDArray<T>(shape_, new_data);
}

template <typename T>
NDArray<T> NDArray<T>::operator-= (const NDArray<T>& arr) {
    // check if the shapes are the same
    if (shape_ != arr.shape_) {
        throw std::invalid_argument("Shapes are not the same");
    }
    for (int i = 0; i < size_; i++) {
        data[i] -= arr.data[i];
    }
    return *this;
}

template <typename T>
NDArray<T> NDArray<T>::operator-= (const T scalar) {
    for (int i = 0; i < size_; i++) {
        data[i] -= scalar;
    }
    return *this;
}


template <typename T>
NDArray<T> NDArray<T>::operator*(const NDArray<T>& arr) {
    // check if the shapes are the same
    if (shape_ != arr.shape_) {
        throw std::invalid_argument("Shapes are not the same");
    }
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] * arr.data[i];
    }
    return NDArray<T>(shape_, new_data);
}

template <typename T>
NDArray<T> NDArray<T>::operator/(const NDArray<T>& arr) {
    // check if the shapes are the same
    if (shape_ != arr.shape_) {
        throw std::invalid_argument("Shapes are not the same");
    }
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] / arr.data[i];
    }
    return NDArray<T>(shape_, new_data);
}

template <typename T>
NDArray<T> NDArray<T>::operator*(T value) {
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] * value;
    }
    return NDArray<T>(shape_, new_data);
}


template <typename T>
NDArray<T> NDArray<T>::operator/(T value) {
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i] / value;
    }
    return NDArray<T>(shape_, new_data);
}

template <typename T>
void NDArray<T>::set(std::vector<int> index, T value) {
    int offset = 0;
    for (int i = 0; i < rank_; i++) {
        offset += index[i] * strides_[i];
    }
    data[offset] = value;
}

template <typename T>
NDArray<T> NDArray<T>::matMult(const NDArray<T>& arr) {
    // Matrix multiplication
    // check if the shapes make sense
    if (shape_[1] != arr.shape_[0]) {
        throw std::invalid_argument("Shapes are not compatible");
    }
    // std::vector<int> new_shape = {shape_[0], arr.shape_[1]};
    // std::vector<T> new_data(new_shape[0] * new_shape[1]);
    // for (int i = 0; i < new_shape[0]; i++) {
    //     for (int j = 0; j < new_shape[1]; j++) {
    //         T sum = 0;
    //         for (int k = 0; k < shape_[1]; k++) {
    //             sum += (*this)[{i, k}] * arr[{k, j}];
    //         }
    //         new_data[i * new_shape[1] + j] = sum;
    //     }
    // }

    // cahche the data
    T* data1 = &data[0];
    T* data2 = (T*)&(arr.data)[0];
    int m = shape_[0];
    int n = arr.shape_[1];
    int k = shape_[1];
    std::vector<T> new_data(m * n);
    T* new_data_ptr = new_data.data();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            T sum = 0;
            for (int l = 0; l < k; l++) {
                sum += data1[i * k + l] * data2[l * n + j];
            }
            new_data_ptr[i * n + j] = sum;
        }
    }
    return NDArray<T>({m, n}, new_data);
}

template <typename T>
NDArray<T> NDArray<T>::tensProd(const NDArray<T>& arr) {
    // Tensor product
    // recursively call the function
    if (rank_ == 1) {
        std::vector<int> new_shape = {shape_[0], arr.shape_[0]};
        std::vector<T> new_data(new_shape[0] * new_shape[1]);
        for (int i = 0; i < new_shape[0]; i++) {
            for (int j = 0; j < new_shape[1]; j++) {
                new_data[i * new_shape[1] + j] = data[i] * arr.data[j];
            }
        }
        return NDArray<T>(new_shape, new_data);
    }
    else {
        std::vector<int> new_shape = {shape_[0], arr.shape_[0]};
        std::vector<T> new_data(new_shape[0] * new_shape[1]);
        NDArray<T> temp = (*this)[0].tensProd(arr[0]);
        for (int i = 0; i < new_shape[0]; i++) {
            for (int j = 0; j < new_shape[1]; j++) {
                temp = (*this)[i].tensProd(arr[j]);
            }
        }
        return temp;
    }
}

template <typename T>
NDArray<T> NDArray<T>::expandDims(int axis) {
    // expand the dimension of the array
    std::vector<int> new_shape = shape_;
    new_shape.insert(new_shape.begin() + axis, 1);
    std::vector<T> new_data(size_);
    for (int i = 0; i < size_; i++) {
        new_data[i] = data[i];
    }
    return NDArray<T>(new_shape, new_data);
}


template <typename T>
std::string NDArray<T>::str(int index) const {
    // recursively print the array depending on the index
    std::stringstream str;
    if (rank_ == 1) {
        str << "[";
        // print data in the array until index
        for (int i = 0; i < data.size(); i++) {
            if(i == data.size() - 1) {
                str << data[i];
            }
            else {
                str << data[i] << ", ";
            }
        }
        str << "]";
    }
    else {
        str << "[";
        for (int i = 0; i < shape_[0]; i++) {
            if (i == shape_[0] - 1) {
                str << (*this)[i].str(index + 1);
            }
            else {
                str << (*this)[i].str(index + 1) << ", ";
            }
        }
        str << "]";
    }
    return str.str();
}

template <typename T>
int NDArray<T>::size() {
    return size_;
}

template <typename T>
int NDArray<T>::size(int dim) {
    return shape_[dim];
}

template <typename T>
int NDArray<T>::rank() {
    return rank_;
}

template <typename T>
std::vector<int> NDArray<T>::shape() {
    return shape_;
}

template <typename T>
std::vector<int> NDArray<T>::strides() {
    return strides_;
}

template <typename T>
void NDArray<T>::reshape(std::vector<int> shape) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    if (size != size_) {
        throw "Shape size does not match array size";
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size) {
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim, T value) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim, T value, bool copy) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim, bool copy) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}



template <typename T>
void NDArray<T>::resize(bool copy) {
    data.resize(size_);
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape, T value) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape, T value, bool copy) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape, bool copy) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}


template <typename T>
void NDArray<T>::resize(int size, bool copy) {
    data.resize(size);
}

template <typename T>
void NDArray<T>::fill(T value) {
    std::fill(data.begin(), data.end(), value);
}

template <typename T>
void NDArray<T>::fill(T value, bool copy) {
    std::fill(data.begin(), data.end(), value);
}

template <typename T>
void NDArray<T>::fill(bool copy) {
    std::fill(data.begin(), data.end(), 0);
}

template <typename T>
void NDArray<T>::copy(NDArray<T> &other) {
    if (other.size_ != size_) {
        throw "Size mismatch";
    }
    data = other.data;
}

template <typename T>
NDArray<T> NDArray<T>::flatten() {
    NDArray<T> result({size_});
    for (int i = 0; i < size_; i++) {
        result.data[i] = data[i];
    }
    return result;
}

template <typename T>
NDArray<T> NDArray<T>::transpose() {
    // 2d transpose
    if (rank_ != 2) {
        throw "Transpose only works on 2d arrays";
    }
    NDArray<T> result = NDArray<T>({shape_[1], shape_[0]});
    for (int i = 0; i < shape_[0]; i++) {
        for (int j = 0; j < shape_[1]; j++) {
            result.data[j * shape_[0] + i] = data[i * shape_[1] + j];
        }
    }
    return result;
}

template <typename T>
NDArray<T> NDArray<T>::transpose(int dim1, int dim2) {
    NDArray<T> result = *this;
    std::swap(result.shape_[dim1], result.shape_[dim2]);
    std::swap(result.strides_[dim1], result.strides_[dim2]);
    return result;
}

template <typename T>
void NDArray<T>::random() {
    // fill with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    for (int i = 0; i < size_; i++) {
        data[i] = dis(gen);
    }
}

template <typename T>
void NDArray<T>::random(T min, T max) {
    // fill with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    for (int i = 0; i < size_; i++) {
        data[i] = dis(gen);
    }
}

template <typename T>
std::vector<T> NDArray<T>::toVector() {
    return data;
}

template <typename T>
T NDArray<T>::dot(NDArray<T> &other) {
    if (size_ != other.size_) {
        throw std::out_of_range("Size mismatch");
    }
    T sum = 0;
    for (int i = 0; i < size_; i++) {
        sum += data[i] * other.data[i];
    }
    return sum;
}

template <typename T>
T NDArray<T>::sum() {
    T sum = 0;
    for (int i = 0; i < size_; i++) {
        sum += data[i];
    }
    return sum;
}

template <typename T>
NDArray<T> NDArray<T>::round() {
    NDArray<T> result = *this;
    for (int i = 0; i < size_; i++) {
        result.data[i] = std::round(data[i]);
    }
    return result;
}

template <typename T>
NDArray<T> NDArray<T>::abs() {
    NDArray<T> result = *this;
    for (int i = 0; i < size_; i++) {
        result.data[i] = std::abs(data[i]);
    }
    return result;
}

template <typename T>
NDArray<T> NDArray<T>::exp() {
    NDArray<T> result = *this;
    for (int i = 0; i < size_; i++) {
        result.data[i] = std::exp(data[i]);
    }
    return result;
}

template <typename T>
NDArray<T> NDArray<T>::pow(int exponent) {
    NDArray<T> result = *this;
    for (int i = 0; i < size_; i++) {
        result.data[i] = std::pow(data[i], exponent);
    }
    return result;
}

template <typename T>
NDArray<T> NDArray<T>::sum(int axis) {
    if (axis >= rank_) {
        throw std::out_of_range("Axis out of range");
    }
    NDArray<T> result = *this;
    result.shape_.erase(result.shape_.begin() + axis);
    result.strides_.erase(result.strides_.begin() + axis);
    result.rank_--;
    result.size_ = result.size_ / shape_[axis];
    for (int i = 0; i < result.size_; i++) {
        result.data[i] = 0;
    }
    for (int i = 0; i < size_; i++) {
        int index = i / strides_[axis] % shape_[axis];
        result.data[i / strides_[axis + 1]] += data[i];
    }
    return result;
}

template <typename T>
NDArray<T> NDArray<T>::inv(){
    NDArray<T> result = *this;
    for (int i = 0; i < size_; i++) {
        result.data[i] = 1 / data[i];
    }
    return result;
}

template <typename T>
NDArray<T> NDArray<T>::eq(NDArray<T> &other) {
    if (size_ != other.size_) {
        throw std::out_of_range("Size mismatch");
    }
    NDArray<T> result = *this;
    for (int i = 0; i < size_; i++) {
        result.data[i] = data[i] == other.data[i];
    }
    return result;
}

template <typename T>
std::string read_shape(NDArray<T> array) {
    std::stringstream os;
    os << "[";
    for (int i = 0; i < array.shape().size(); i++) {
        os << array.shape()[i];
        if (i != array.shape().size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os.str();
}

template <typename T>
using ndarray = NDArray<T>;

#endif