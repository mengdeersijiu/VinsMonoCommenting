#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;//如果size==7 return6; 如果size不为7 return size
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;

    //添加残差块相关信息（优化变量，待marg的变量）
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    //计算每个残差对应的雅克比，并更新parameter_block_data
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;//所有观测项

    int m, n;//m为需要marg掉的变量，n为需要保留的变量
    std::unordered_map<long, int> parameter_block_size; //global size <优化变量内存地址,各个优化变量的长度>
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size <优化变量内存地址，各个优化变量在矩阵中的id>
    std::unordered_map<long, double *> parameter_block_data;//<优化变量内存地址,数据>

    std::vector<int> keep_block_size; //global size 边缘化之后保留下来的各个优化变量的长度
    std::vector<int> keep_block_idx;  //local size 边缘化之后保留下来的各个优化变量在id
    std::vector<double *> keep_block_data; //边缘化之后保留下来的各个优化变量对应的double指针类型的数据

    Eigen::MatrixXd linearized_jacobians; //边缘化之后从信息矩阵恢复出来的雅克比矩阵
    Eigen::VectorXd linearized_residuals; //边缘化之后从信息矩阵恢复出来的残差向量
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
