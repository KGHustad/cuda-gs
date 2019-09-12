import os
import time
import ctypes
from ctypes import c_int, c_long, c_ulong, c_double, c_void_p
import subprocess

import numpy as np

float64_array_2d = np.ctypeslib.ndpointer(dtype=c_double, ndim=2,
                                          flags="contiguous")


def _load_lib(rebuild=False):
    lib_filename = "build/lib/libgs_gpu.so"
    libdir = os.path.dirname(__file__)
    if rebuild:
        args = ['make', '-C', libdir, '-f', 'Makefile.Python', lib_filename]
        cp = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if cp.returncode != 0:
            print(cp.stdout.decode('utf-8'))
            print(cp.stderr.decode('utf-8'))
            raise ImportError("Failed to build CUDA library")
    lib_path = os.path.join(libdir, lib_filename)
    if not os.path.isfile(lib_path):
        raise ImportError("CUDA library was not found")
    lib = np.ctypeslib.load_library(lib_path, '.')
    has_nccl = _setup_functions(lib)
    return lib, has_nccl

def _setup_functions(lib):
    _setup_function_gs_init(lib.gs_init)
    _setup_function_gs_cleanup(lib.gs_cleanup)
    _setup_function_gs_copy_from_host(lib.gs_copy_from_host)
    _setup_function_gs_copy_to_host(lib.gs_copy_to_host)
    _setup_function_gs_orthogonalise_vector(lib.gs_orthogonalise_vector)
    has_nccl = False
    try:
        _setup_function_gs_init(lib.gs_init_nccl)
        _setup_function_gs_cleanup(lib.gs_cleanup_nccl)
        _setup_function_gs_copy_from_host(lib.gs_copy_from_host_nccl)
        _setup_function_gs_copy_to_host(lib.gs_copy_to_host_nccl)
        _setup_function_gs_orthogonalise_vector(lib.gs_orthogonalise_vector_nccl)
        has_nccl = True
    except AttributeError:
        pass
    return has_nccl

def _setup_function_gs_init(c_func):
    c_func.restype = c_int # return code for success/failure
    c_func.argtypes = [
        c_void_p,       # gs_data
        c_int,          # M
        c_int,          # N
    ]

    def errcheck(return_code, func, params):
        init_return_code_desc = [
            'success',
            'no GPU devices found',
            'insuffient device memory',
            'missing OpenMP',
        ]
        if return_code != 0:
            msg = "ERROR: {0}".format(init_return_code_desc[return_code])
            raise RuntimeError(msg)

    c_func.errcheck = errcheck

def _setup_function_gs_cleanup(c_func):
    c_func.restype = None # void
    c_func.argtypes = [
        c_long,          # gs_data
    ]

def _setup_function_gs_copy_from_host(c_func):
    c_func.restype = None # void
    c_func.argtypes = [
        c_long,           # gs_data
        float64_array_2d, # V
    ]

def _setup_function_gs_copy_to_host(c_func):
    c_func.restype = None # void
    c_func.argtypes = [
        c_long,           # gs_data
        float64_array_2d, # V
    ]

def _setup_function_gs_orthogonalise_vector(c_func):
    c_func.restype = None # void
    c_func.argtypes = [
        c_long,           # gs_data
        c_int,            # new_vec_ind
    ]

_lib, has_nccl = _load_lib()

def log_timestamp(timestamp_start, desc):
    time_elapsed = time.time() - timestamp_start
    print("[{0:6.3f} s]: {1}".format(time_elapsed, desc))

def log_event(timestamp_start, desc, events):
    timestamp_prev = timestamp_start
    if len(events) > 0:
        timestamp_prev += events[-1][1]
    time_elapsed = time.time() - timestamp_start
    event_time = time.time() - timestamp_prev

    event = (desc, time_elapsed, event_time)
    events.append(event)

def orthogonalise(V, verbose=False):
    timestamp_start = time.time()
    assert len(V.shape) == 2
    M, N = V.shape
    if M >= N:
        print ("Warning: M >= N, {0} >= {1}.".format(M, N))

    # we need 8 bytes to hold a pointer to the gs_data struct
    data_placeholder = c_long(0)
    # data_ptr will be used as a pointer to pointer
    data_ptr = ctypes.byref(data_placeholder)

    device_count = c_int(0)
    device_count_ptr = ctypes.byref(device_count)

    events = []

    _lib.gs_init(data_ptr, M, N, device_count_ptr)
    log_event(timestamp_start, "init", events)
    if verbose:
        #print("init done")
        log_timestamp(timestamp_start, "init done")

    _lib.gs_copy_from_host(data_placeholder, V)
    log_event(timestamp_start, "copy from host", events)
    if verbose:
        #print("copy from host done")
        log_timestamp(timestamp_start, "copy from host done")

    timestamp_orthogonalisation_start = time.time()
    for new_vec_ind in range(1, M):
        _lib.gs_orthogonalise_vector(data_placeholder, new_vec_ind)
    timestamp_orthogonalisation_done = time.time()
    log_event(timestamp_start, "orthogonalisation", events)
    if verbose:
        time_elapsed_orthogonalisation = timestamp_orthogonalisation_done - timestamp_orthogonalisation_start
        desc = "orthogonalisation done in {0:g} seconds".format(time_elapsed_orthogonalisation)
        log_timestamp(timestamp_start, desc)

    _lib.gs_copy_to_host(data_placeholder, V)
    log_event(timestamp_start, "copy to host", events)
    if verbose:
        log_timestamp(timestamp_start, "copy to host done")


    _lib.gs_cleanup(data_placeholder)
    log_event(timestamp_start, "cleanup", events)
    if verbose:
        log_timestamp(timestamp_start, "cleanup done")

    return events

def orthogonalise_nccl(V, verbose=False):
    if not has_nccl:
        raise RuntimeError("NCCL functions were not built and cannot be called")
    timestamp_start = time.time()
    assert len(V.shape) == 2
    M, N = V.shape
    if M >= N:
        print ("Warning: M >= N, {0} >= {1}.".format(M, N))

    # we need 8 bytes to hold a pointer to the gs_data struct
    data_placeholder = c_long(0)
    # data_ptr will be used as a pointer to pointer
    data_ptr = ctypes.byref(data_placeholder)

    device_count = c_int(0)
    device_count_ptr = ctypes.byref(device_count)

    events = []

    _lib.gs_init_nccl(data_ptr, M, N, device_count_ptr)
    log_event(timestamp_start, "init", events)
    if verbose:
        log_timestamp(timestamp_start, "init done")
    print("Using {0} devices".format(device_count.value))

    _lib.gs_copy_from_host_nccl(data_placeholder, V)
    log_event(timestamp_start, "copy from host", events)
    if verbose:
        log_timestamp(timestamp_start, "copy from host done")

    timestamp_orthogonalisation_start = time.time()
    for new_vec_ind in range(1, M):
        _lib.gs_orthogonalise_vector_nccl(data_placeholder, new_vec_ind)
    timestamp_orthogonalisation_done = time.time()
    log_event(timestamp_start, "orthogonalisation", events)
    if verbose:
        time_elapsed_orthogonalisation = timestamp_orthogonalisation_done - timestamp_orthogonalisation_start
        desc = "orthogonalisation done in {0:g} seconds".format(time_elapsed_orthogonalisation)
        log_timestamp(timestamp_start, desc)

    _lib.gs_copy_to_host_nccl(data_placeholder, V)
    log_event(timestamp_start, "copy to host", events)
    if verbose:
        log_timestamp(timestamp_start, "copy to host done")

    _lib.gs_cleanup_nccl(data_placeholder)
    log_event(timestamp_start, "cleanup", events)
    if verbose:
        log_timestamp(timestamp_start, "cleanup done")

    return events
