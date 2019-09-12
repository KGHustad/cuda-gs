import time
import numpy as np

has_numba = False
try:
    import numba
    has_numba = True
except ImportError:
    pass

has_cupy = False
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    pass

import gs_c_lib

has_gs_cuda_lib = False
try:
    import gs_cuda_lib
    has_gs_cuda_lib = True
except ImportError:
    pass


def gs_numpy(V):
    for j in range(1,N):
        inner_prods_uv = np.sum(V[:,j,None]*V[:,:j], axis=0)
        inner_prods_uu= np.sum(V[:,:j]*V[:,:j],axis=0)
        V[:,j] -= np.sum(inner_prods_uv/inner_prods_uu*V[:,:j],axis=1)

def gs_numpy_T(V):
    for j in range(1,N):
        inner_prods_uv = np.sum(V[j,:]*V[:j,:], axis=1)
        inner_prods_uu= np.sum(V[:j,:]*V[:j,:], axis=1)
        V[j,:] -= np.sum((inner_prods_uv/inner_prods_uu)[:,np.newaxis]*V[:j,:],axis=0)

if has_numba:
    @numba.jit(cache=True, nopython=True)
    def gs_numba(V):
        for j in range(1,N):
            for i in range(j):
                inner_prod_uv = np.sum(V[:,j]*V[:,i], axis=0)
                inner_prod_uu= np.sum(V[:,i]*V[:,i],axis=0)
                V[:,j] -= inner_prod_uv/inner_prod_uu*V[:,i]

    @numba.jit(cache=True, nopython=True)
    def gs_numba_T(V):
        for j in range(1,N):
            for i in range(j):
                inner_prod_uv = np.dot(V[j,:],V[i,:])
                inner_prod_uu= np.dot(V[i,:],V[i,:])
                V[j,:] -= (inner_prod_uv/inner_prod_uu)*V[i,:]

    @numba.jit(cache=True, nopython=True)
    def gs_numba_T_plain(V, tol=1E-14):
        M, N = V.shape
        tol_squared = tol*tol
        for new_vec_ind in range(1,M):
            for i in range(new_vec_ind):
                inner_prod_uu = 0
                inner_prod_uv = 0
                for k in range(N):
                    inner_prod_uu += V[i,k]*V[i,k]
                    inner_prod_uv += V[i,k]*V[new_vec_ind,k]
                if inner_prod_uu > tol_squared:
                    fac = inner_prod_uv / inner_prod_uu
                    for k in range(N):
                        V[new_vec_ind,k] -= fac * V[i,k]

def gs_cupy(V):
    for j in range(1,N):
        inner_prods_uv = cp.sum(V[:,j,None]*V[:,:j], axis=0)
        inner_prods_uu= cp.sum(V[:,:j]*V[:,:j],axis=0)
        V[:,j] -= cp.sum(inner_prods_uv/inner_prods_uu*V[:,:j],axis=1)

def gs_cupy_T(V):
    for j in range(1,N):
        inner_prods_uv = cp.sum(V[j,:]*V[:j,:], axis=1)
        inner_prods_uu = cp.sum(V[:j,:]*V[:j,:],axis=1)
        V[j,:] -= cp.sum((inner_prods_uv/inner_prods_uu)[:,np.newaxis]*V[:j,:],axis=0)


def gs_C_T(V):
    M, N = V.shape
    assert M < N, "M={0}, N={1}".format(M, N)
    for j in range(1, M):
        gs_c_lib.orthogonalise_vector(V, j)

def gs_C_omp_T(V):
    M, N = V.shape
    assert M < N, "M={0}, N={1}".format(M, N)
    for j in range(1, M):
        gs_c_lib.orthogonalise_vector_omp(V, j)

def gs_cuda_T(V, verbose=False):
    M, N = V.shape
    assert M < N, "M={0}, N={1}".format(M, N)
    events = gs_cuda_lib.orthogonalise(V, verbose=verbose)

    time_spent_orthogonalisation = 0
    for desc, total_elapsed, event_time in events:
        if desc == 'orthogonalisation':
            time_spent_orthogonalisation = event_time
            break
    #time_spent_orthogonalisation = events['orthogonalisation'][2]
    return time_spent_orthogonalisation

def gs_cuda_nccl_T(V, verbose=False):
    M, N = V.shape
    assert M < N, "M={0}, N={1}".format(M, N)
    events = gs_cuda_lib.orthogonalise_nccl(V, verbose=verbose)

    time_spent_orthogonalisation = 0
    for desc, total_elapsed, event_time in events:
        if desc == 'orthogonalisation':
            time_spent_orthogonalisation = event_time
            break
    #time_spent_orthogonalisation = events['orthogonalisation'][2]
    return time_spent_orthogonalisation

def check_ort(A):
    print(np.dot(A[:,0], A[:,1]), np.dot(A[:,0], A[:,2]), np.dot(A[:,1], A[:,2]))

def check_ort_cp(A):
    print(cp.dot(A[:,0], A[:,1]), cp.dot(A[:,0], A[:,2]), cp.dot(A[:,1], A[:,2]))

def check_ort_cp_T(A):
    print(cp.dot(A[0,:], A[1,:]), cp.dot(A[0,:], A[2,:]), cp.dot(A[1,:], A[2,:]))


def bench_gs_numpy(mem_traffic=None):
    method_desc = "numpy"
    V = V_orig.copy()
    pre = time.time()
    gs_numpy(V)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V)
    print()
    return time_taken

def bench_gs_numpy_T(mem_traffic=None):
    method_desc = "numpy transposed"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_numpy_T(V_T)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

def bench_gs_numba(mem_traffic=None):
    method_desc = "numba"
    V = V_orig.copy()
    pre = time.time()
    gs_numba(V)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V)
    print()
    return time_taken

def bench_gs_numba_T(mem_traffic=None):
    method_desc = "numba transposed"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_numba_T(V_T)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

def bench_gs_numba_T_plain(mem_traffic=None):
    method_desc = "numba transposed plain loops"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_numba_T_plain(V_T)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

def bench_gs_cupy(mem_traffic=None):
    method_desc = "cupy"
    V_d = cp.array(V_orig.copy())
    pre = time.time()
    gs_cupy(V_d)
    check_ort_cp(V_d)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    print()
    return time_taken

def bench_gs_cupy_T(mem_traffic=None):
    method_desc = "cupy transposed"
    V_T_d = cp.array(V_orig.T.copy())
    pre = time.time()
    gs_cupy_T(V_T_d)
    check_ort_cp_T(V_T_d)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    print()
    return time_taken

def bench_gs_C_T(mem_traffic=None):
    method_desc = "C transposed"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_C_T(V_T)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

def bench_gs_C_omp_T(mem_traffic=None):
    method_desc = "C with OpenMP transposed"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_C_omp_T(V_T)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

def bench_gs_cuda_T(mem_traffic=None, verbose=False):
    method_desc = "CUDA"
    V_T = V_orig.T.copy()
    #pre = time.time()
    time_taken = gs_cuda_T(V_T, verbose=verbose)
    #post = time.time()
    #time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

def bench_gs_cuda_nccl_T(mem_traffic=None, verbose=False):
    method_desc = "multi-GPU CUDA with NCCL"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_cuda_nccl_T(V_T, verbose=verbose)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

if has_numba:
    @numba.njit(cache=True)
    def _generate_V(M, N):
        np.random.seed(1)
        V = np.random.rand(M, N)
        return V

    def generate_V(M, N):
        pre = time.time()
        V = _generate_V(M, N)
        post = time.time()
        if post - pre > 1:
            print("Generated random V in {0:g} seconds".format(post-pre))
        return V
else:
    def generate_V(M, N):
        pre = time.time()
        np.random.seed(1)
        V = np.random.rand(M, N)
        post = time.time()
        if post - pre > 1:
            print("Generated random V in {0:g} seconds".format(post-pre))
        return V



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', "--length", type=float, default=1E6,
            help="length of each vector")
    parser.add_argument('-N', "--vectors", type=float, default=300,
            help="number of vectors to orthogonalise")
    args = parser.parse_args()

    M = int(args.length)
    N = int(args.vectors)
    assert M > 0
    assert N > 00

    sizeof_double = 8
    print("(M, N): {0}, array size: {1:g} MB".format(
        (M, N), M*N*sizeof_double/1E6))

    V_orig = generate_V(M, N)

    # the CPU code reads the vector being orthogonalised more often, leading to higher memory traffic
    mem_traffic_optimal = (N*N + N*N/2. + 2*N)*M*sizeof_double
    mem_traffic_optimal_cuda = (N*N + 2*N)*M*sizeof_double
    bytes_two_vectors = 2*M*sizeof_double
    print("Optimal memory traffic is {0:g} GB, assuming {1:.0f} MB cannot fit in cache".format(
        mem_traffic_optimal/1E9,
        bytes_two_vectors/1E6,
    ))
    print()
    if mem_traffic_optimal < 100E9:
        bench_gs_numpy()
        bench_gs_numpy_T()
        pass
    else:
        print("Skipping numpy benchmark since it will take forever")
    print()
    if has_numba:
        if mem_traffic_optimal < 500E9:
            #bench_gs_numba()
            bench_gs_numba_T(mem_traffic=mem_traffic_optimal)
            bench_gs_numba_T_plain(mem_traffic=mem_traffic_optimal)
        else:
            print("Skipping numba benchmark")
        print()

    if mem_traffic_optimal < 1000E9:
        if mem_traffic_optimal < 500E9:
            bench_gs_C_T(mem_traffic=mem_traffic_optimal)
        else:
            print("Skipping serial C benchmark since it will take very long")

        bench_gs_C_omp_T(mem_traffic=mem_traffic_optimal)
    else:
        print("Skipping CPU benchmarks since they will take very long")
    print()

    if has_gs_cuda_lib:
        bench_gs_cuda_T(mem_traffic=mem_traffic_optimal_cuda, verbose=True)
        if gs_cuda_lib.has_nccl:
            bench_gs_cuda_nccl_T(mem_traffic=mem_traffic_optimal_cuda, verbose=True)
        print()

    if has_cupy:
        bench_gs_cupy()
        bench_gs_cupy_T()
        bench_gs_cupy_jonas()
