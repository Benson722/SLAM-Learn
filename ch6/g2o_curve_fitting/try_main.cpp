#include "iostream"
#include "Eigen/Core"
#include "opencv2/core/core.hpp"
#include "cmath"
#include "chrono"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_dogleg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"

using namespace std;

//Vertex of Curve Function, Template Parameters: Optimise Para dimension and type
class CurveFittingVertex: public g2o::BaseVertex<4, Eigen::Vector4d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() //Reset
    {
        _estimate << 0, 0, 0, 0;
    }

    virtual void oplusImpl( const double* update) //Update
    {
        _estimate += Eigen::Vector4d(update);
    }

    //Read and Write
    virtual bool read(istream& in){ }
    virtual bool write( ostream& out) const { }
};

//Cost Model.  Template: Dimensions, Type, Vertex Type
class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge( double x ): BaseUnaryEdge(), _x(x) {}
    //Compute Error
    void computeError()
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector4d abcd = v ->estimate();
        _error(0,0) = _measurement - std::exp( abcd(0,0) * _x * _x * _x+ abcd(1,0) * _x * _x+ abcd(2,0) * _x + abcd(3,0));
    }
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
public:
    double _x;
};

int main( int argc, char** argv)
{
    double a_true = 1.0, b_true = 2.0, c_true = 3.0, d_true = 4.0;
    int Nums = 100;
    double w_sigma = 1.0;
    cv::RNG rng;
    double abcd[4] = {0, 0, 0, 0 };

    vector<double> x_data, y_data ; 

    cout << "Generate Data ..." << endl;
    for( int num = 0 ; num < Nums ; num ++)
    {
        double x = num / 100.0;
        x_data.push_back( x );
        y_data.push_back(
            std::exp(a_true * x * x * x + b_true * x * x + c_true * x + d_true) + rng.gaussian(w_sigma * w_sigma)
        );
        cout << x_data[num] << " & " << y_data[num] << endl;
    }

    //Construct Graph Optimism
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<4,1>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block( linearSolver );
    //Gradient Way: G-N, Levenberg, DogLeg
    // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm( solver);
    optimizer.setVerbose( true) ;

    //Add Vertex
    CurveFittingVertex* v = new CurveFittingVertex();
    v -> setEstimate(Eigen::Vector4d(0, 0, 0, 0));
    v -> setId(0);
    optimizer.addVertex( v );

    //Add Edge
    for (int i = 0;  i < Nums ; i++)
    {
        CurveFittingEdge* edge = new CurveFittingEdge( x_data[i]);
        edge->setId(i);
        edge->setVertex( 0, v);
        edge->setMeasurement( y_data[i]);
        edge->setInformation( Eigen::Matrix<double, 1 , 1>::Identity() * 1/(w_sigma * w_sigma));
        optimizer.addEdge( edge);
    }

    //Run g2o!
    cout << "Start Optimization ! " << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
    
    // 输出优化值
    Eigen::Vector4d abcd_estimate = v->estimate();
    cout<<"estimated model: "<<abcd_estimate.transpose()<<endl;
    return 0;
}