
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include <stdio.h>

using namespace flann;

int main(int argc, char** argv)
{

	if (argc < 5)
	{
		printf("Usage: %s <filein1> <name1> <filein2> <name2> <fileout> \n", argv[0]);
		printf("example: %s %s %s %s %s %s\n",  argv[0], "dataset.hdf5","dataset", "dataset.hdf5","query", "result.hdf5");
		exit(1);
	}

	std::string f1, n1, f2, n2, fo;
	f1 = argv[1];
	n1 = argv[2];
	f2 = argv[3];
	n2 = argv[4];

	fo = argv[5];


	Matrix<float> dataset;
    Matrix<float> query;

    //load_from_file(dataset, "dataset.hdf5","dataset");
    //load_from_file(query, "dataset.hdf5","query");

	load_from_file(dataset, f1, n1);
    load_from_file(query, f2, n2);


    int nn = 1;
    Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
    Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

    // construct an randomized kd-tree index using 4 kd-trees
    Index<L2<float> > index(dataset, flann::KDTreeIndexParams(8));
    index.buildIndex();                                                                                               

    // do a knn search, using 128 checks
    index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

    flann::save_to_file(indices, fo, "result");
    flann::save_to_file(dists, fo, "dists");

	Matrix<int> cc(new int[1], 1, 1);
	*cc[0] = index.getCompCount();
   
	flann::save_to_file(cc, fo, "compCount");


#define show(x) std::cout << #x << " : " << x << std::endl;
	
	show(dataset.rows);
	show(dataset.cols);
	show(query.rows);
	
	show(index.getCompCount());


    delete[] dataset.ptr();
    delete[] query.ptr();
    delete[] indices.ptr();
    delete[] dists.ptr();
    
    return 0;
}
