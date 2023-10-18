__kernel void hello_kernel(__global const int *a,
                           __global const int *b,
                           __global int *result_matrix,int  result_matrix_row,
                           int  result_matrix_col,int  compute_size)
{
    int row = get_global_id(0);
    int col = get_global_id(1);


    int sum = 0;
    for(int i=0;i<compute_size;i++)
    {
        sum += a[row*compute_size+i] * b[i*result_matrix_col+col];
    }

    result_matrix[row*result_matrix_col+col] = sum;
}