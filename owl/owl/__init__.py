#!/usr/bin/env python
""" Owl module: Python binding for Minerva library

The module encapsulates or directly maps some functions of Minerva's API to python.
Only system APIs are defined here. For convolution APIs, please refer to conv.py.
For element-wise operations, please refer to elewise.py. For other APIs such as member
functions of :py:class:`owl.NArray`, please refer to the API document.

Note that Minerva is an dataflow system with lazy evaluation to construct dataflow graph
to extract parallelism within codes. All the operations like ``+-*/`` of ``owl.NArray`` are
all *lazy* such that they are NOT evaluated immediatedly. Only when you try to pull the
content of an NArray outside Minerva's control (e.g. call ``to_numpy()``) will those operations
be executed concretely.

We implement two interfaces for swapping values back and forth between numpy and Minerva.
They are ``from_numpy`` and ``to_numpy``. So you could still use any existing codes
on numpy such as IO and visualization.
"""
import numpy as np
import libowl as _owl
import sys
import time
NArray = _owl.NArray
_owl.initialize()


"""
worker nodes drop into forever loop.
"""
if _owl.has_mpi() and _owl.rank() != 0:
    _owl.WorkerRun()


# def initialize():
#     """ Initialize Minerva System with `sys.argv`
#
#     .. note::
#         Must be called before calling any owl's API
#     """
#     _owl.initialize()

def has_cuda():
    """ Check if CUDA is enabled

    :return: CUDA status
    :rtype: int
    """
    return _owl.has_cuda()

def has_mpi():
    """ Check if MPI is enabled
    
    :return MPI status
    :rtype int
    """
    return _owl.has_mpi()

def rank():
    """ The MPI rank of this process
    
    :return MPI rank
    :rtype int
    """
    return _owl.rank()

def worker_run():
    """ The main loop of the mpi worker 
    
    :return none
    """
    return _owl.WorkerRun()

def wait_for_all():
    """ Wait for all evaluation to complete

    .. note::
        The user thread (python) will be blocked until all previous operations are finished.

    :return: None
    """
    _owl.wait_for_all()

def create_cpu_device():
    """ Create device for running on CPU cores

    .. note::
        At least one of :py:func:`create_cpu_device` or :py:func:`create_gpu_device` should be called
        before using any ``owl`` APIs.

    :return: A unique id for cpu device
    :rtype: int
    """
    return _owl.create_cpu_device()

def create_gpu_device(which):
    """ Create device for running on GPU card

    .. note::
        At least one of :py:func:`create_cpu_device` or :py:func:`create_gpu_device` should be called
        before using any ``owl`` APIs.

    :param int which: which GPU card the code would be run on
    :return: A unique id for the device on that GPU card
    :rtype: int
    """
    return _owl.create_gpu_device(which)

def create_mpi_device(rank, which):
    """ Create device for running on GPU card

    .. note::
        At least one of :py:func:`create_cpu_device` or :py:func:`create_gpu_device` should be called
        before using any ``owl`` APIs.

    :param int which: which GPU card the code would be run on
    :return: A unique id for the device on that GPU card
    :rtype: int
    """
    return _owl.create_mpi_device(rank, which)

def get_gpu_device_count():
    """ Get the number of compute-capable GPU devices

    :return: Number of compute-capable GPU devices
    :rtype: int
    """
    return _owl.get_gpu_device_count()

def get_mpi_node_count():
    """ Get the number of MPI connected compute nodes

    :return: Number of MPI compute nodes
    :rtype: int
    """
    return _owl.get_mpi_node_count()


def get_mpi_device_count(r):
    """ Get the number of compute-capable GPU devices on a given mpi node

    :param int rank: The rank of the mpi node
    :return: Number of compute-capable GPU devices on given mpi node
    :rtype: int
    """
    return _owl.get_mpi_device_count(r)


def set_device(dev):
    """ Switch to the given device for running computations

    When ``set_device(dev)`` is called, all the subsequent codes will be run on ``dev``
    till another ``set_device`` is called.

    :param int dev: the id of the device (usually returned by create_xxx_device)
    """
    _owl.set_device(dev)

def zeros(shape):
    """ Create ndarray of zero values

    :param shape: shape of the ndarray to create
    :type shape: list int
    :return: result ndarray
    :rtype: owl.NArray
    """
    return NArray.zeros(shape)

def ones(shape):
    """ Create ndarray of one values

    :param shape: shape of the ndarray to create
    :type shape: list int
    :return: result ndarray
    :rtype: owl.NArray
    """
    return NArray.ones(shape)

def randn(shape, mu, var):
    """ Create a random ndarray using normal distribution

    :param shape: shape of the ndarray to create
    :type shape: list int
    :param float mu: mu
    :param float var: variance
    :return: result ndarray
    :rtype: owl.NArray
    """
    return NArray.randn(shape, mu, var)

def randb(shape, prob):
    """ Create a random ndarray using bernoulli distribution

    :param shape: shape of the ndarray to create
    :type shape: list int
    :param float prob: probability for the value to be one
    :return: result ndarray
    :rtype: owl.NArray
    """
    return NArray.randb(shape, prob)

def from_numpy(nparr):
    """ Create an owl.NArray from numpy.ndarray

    .. note::

        The content will be directly copied to Minerva's memory system. However, due to
        the different priority when treating dimensions between numpy and Minerva. The
        result ``owl.NArray``'s dimension will be *reversed*.

        >>> a = numpy.zeros([200, 300, 50])
        >>> b = owl.from_numpy(a)
        >>> print b.shape
        [50, 300, 200]

    .. seealso::

        :py:func:`owl.NArray.to_numpy`

    :param numpy.ndarray nparr: numpy ndarray
    :return: Minerva's ndarray
    :rtype: owl.NArray
    """
    return NArray.from_numpy(np.require(nparr, dtype=np.float32, requirements=['C']))

def concat(narrays, concat_dim):
    """  Concatenate NArrays according to concat_dim

    :param narrays: inputs for concatenation
    :type narrays: owl.NArray
    :param concat_dim: the dimension to concate
    :type concat_dim: int

    :return: result of concatenation
    :rtype: owl.NArray
    """
    return NArray.concat(narrays, concat_dim)

def slice(src, slice_dim, st_off, slice_count):
    """  Slice NArrays according to slice_dim

    :param src: inputs for slice
    :type src: owl.NArray
    :param slice_dim: the dimension to slice
    :type slice_dim: int
    :param st_off: where to start slice
    :type st_off: int
    :param slice_count: how many data_chunk on slice_dim
    :slice_count: int

    :return: result of slicer
    :rtype: owl.NArray
    """
    return NArray.slice(src, slice_dim, st_off, slice_count)

# def print_profiler_result():
#     """ Print result from execution profiler
#
#     :return: None
#     """
#     _owl.print_profiler_result()
#
# def reset_profiler_result():
#     """ Reset execution profiler
#
#     :return: None
#     """
#     _owl.reset_profiler_result()
#
# def print_dag_to_file(fname):
#     """ Print the current generated dag into the give filename
#
#     :param fname: filename for printing the dag
#     :type fname: str
#     :return: None
#     """
#     _owl.print_dag_to_file(fname)
#
# def print_dot_dag_to_file(fname):
#     """ Print the current generated dag into the give filename in dot format
#
#     :param fname: filename for printing the dag
#     :type fname: str
#     :return: None
#     """
#     _owl.print_dot_dag_to_file(fname)
