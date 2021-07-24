#include <iostream>
#include <mpi.h>
#include <fstream>
std::ofstream out;

int main(int argc, char* argv[]){
        int ierr = MPI_Init(&argc, &argv);
        int procid, P;
        out.open("/home/ubuntu/mpi.txt");
        ierr = MPI_Comm_rank(MPI_COMM_WORLD,&procid);
        ierr = MPI_Comm_size(MPI_COMM_WORLD,&P);
	out << "COMM SIZE: " << P << std::endl;
	out << "procid: " << procid << std::endl;
	std::cout << "procid: " << procid << std::endl;
        ierr = MPI_Finalize();
        out.close();
        return 0;
}
