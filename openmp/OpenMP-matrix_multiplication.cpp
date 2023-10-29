#include <iostream>
#include <pthread.h>
#include <omp.h>
#include <sys/time.h>
#include <ctime>
using namespace std;
const int maxn = 4;

int A[maxn][maxn], B[maxn][maxn], C[maxn][maxn];

int main() {
    int i, j, k;

    omp_set_num_threads(omp_get_num_procs());
    srand(time(NULL));
    for (i = 0; i < maxn; i++)
        for (j = 0; j < maxn; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }

    #pragma omp parallel for private(i,j,k) shared(A,B,C)
    for (i = 0; i < maxn; ++i)
        for (j = 0; j < maxn; ++j)
            for (k = 0; k < maxn; ++k){
                //printf("OpenMP Test, : %d\n", omp_get_thread_num());
                C[i][j] += A[i][k] * B[k][j];
            }
                

    for (i = 0; i < maxn; i++) {
        for (j = 0; j < maxn; j++)
            cout << C[i][j] << "\t";
        cout << endl;
    }
}
