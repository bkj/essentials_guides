# PageRank: From `networkx` to `gunrock/essentials`

`pagerank` is an algorithm for computing a ranking of the nodes of a graph based on link structure.  It's famous for being the (original) basis of the Google search algorithm.

This document walks through the process of going from low-performance `pagerank` implementations available in commonly used Python libraries, to higher-performance implementations using the Python `numba` library and C++, to very high performance implementations using the `essentials` GPU graph processing framework.

## NetworkX Baselines

`networkx` has a few different `pagerank` implementations, visible [here](https://github.com/networkx/networkx/blob/main/networkx/algorithms/link_analysis/pagerank_alg.py)  `pagerank` can be framed from either a "graph" perspective (operations on nodes and edges) or a "linear algebra" perspective (operations on matrices and vectors).  

The [fastest implementation](https://github.com/networkx/networkx/blob/main/networkx/algorithms/link_analysis/pagerank_alg.py#L357) in `networkx` uses the linear algebra approach, but they also have a [graph-based implementation](https://github.com/networkx/networkx/blob/main/networkx/algorithms/link_analysis/pagerank_alg.py#L112).

Here are simplified versions of both implementations (with some code for directed graphs and special cases removed).  `pagerank_graph` is the graph-centric implementation, `pagerank_alg` is the linear-algebra-centric implementation.

Note: for `pagerank_alg`, I pass the adjacency matrix as a `scipy.sparse` matrix, to avoid unnecessary format conversions, which are expensive.

```python
def pagerank_python(G, alpha=0.85, tol=1.0e-6):
    """ graph-centric implementation """
    
    D = G.to_directed()
    W = nx.stochastic_graph(D, weight='weight')
    N = W.number_of_nodes()
    
    x = dict.fromkeys(W, 1.0 / N)
    p = dict.fromkeys(W, 1.0 / N)
    
    while True:
        xlast = x
        x     = dict.fromkeys(xlast.keys(), 0)
        
        for n in x:
            for _, nbr, wt in W.edges(n, data='weight'):
                x[nbr] += alpha * xlast[n] * wt
            
            x[n] += (1.0 - alpha) * p.get(n, 0)
        
        err = max([abs(x[n] - xlast[n]) for n in x])
        if err < tol:
            return np.array(list(x.values()))


def pagerank_scipy(adj, alpha=0.85, tol=1e-6):
    """ linear-algebra-centric implementation """
    
    n_nodes   = adj.shape[0]
    S         = np.array(adj.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q         = sparse.spdiags(S.T, 0, *adj.shape, format="csr")
    adj       = Q * adj
    
    x = np.repeat(1.0 / n_nodes, n_nodes)
    p = np.repeat(1.0 / n_nodes, n_nodes)
    
    while True:
        xlast = x
        x     = alpha * (x * adj) + (1 - alpha) * p
        err   = np.abs(x - xlast).max()
        if err < tol:
            return x
```

We can benchmark the runtime of these implementations to understand the kind of performance you could expect from `networkx`.  We'll use the synthetic RMAT-18 graph, which has 174K nodes and 7.6M edges.

```python
adj = mmread('rmat18.mtx')
print(adj.shape[0], adj.nnz)
# 174147 7600696

# --
# `pagerank_graph` -- graph-centric implementation

# convert adjacency matrix to networkx format
t                 = perf_counter_ns()
G                 = nx.from_scipy_sparse_matrix(adj)
timing['convert'] = (perf_counter_ns() - t) / 1e6

# run pagerank_graph
t                 = perf_counter_ns()
x_python          = pagerank_graph(G)
timing['python']  = (perf_counter_ns() - t) / 1e6

# --
# `pagerank_alg` -- linear-algebra-centric implementation

t                 = perf_counter_ns()
x_scipy           = pagerank_alg(adj)
timing['scipy']   = (perf_counter_ns() - t) / 1e6
```

The results are:
```python
{
  'read'    : 5319., 
  'convert' : 55972, 
  'python'  : 440489, 
  'scipy'   : 796,
}
```

So it takes
 - 55s + 440s = 495s = 8.25m to create the `networkx.Graph` and run `pagerank_graph`
 - 945ms to run `pagerank_alg`

Clearly `pagerank_alg` is much faster (by ~ 500x) -- which is presumably why it's now the default `networkx` implementation.

How can we do better?

## Better Python Implementations w/ Numba

The `numba` Python packages offers a relatively easy way to get a performance boost.  We can re-write `pagerank_graph` using `numba` as shown below:

```python
@njit()
def _row_normalize(n_nodes, n_edges, colptr, rindices, data):
  """
    Compute row-normalized adjacency matrix
    Equivalent to `W = nx.stochastic_graph(D, weight='weight')`
  """
  
  # compute "inverse outgoing weight" from each node
  nrms = np.zeros(n_nodes)
  for dst in range(n_nodes):
    for offset in range(colptr[dst], colptr[dst + 1]):
        src = rindices[offset]
        val = data[offset]
        nrms[src] += val
  
  nrms[nrms != 0] = 1 / nrms[nrms != 0]
  print(nrms[:10])
  
  # compute normalized edge weights
  ndata = np.zeros(n_edges)
  for dst in range(n_nodes):
    for offset in range(colptr[dst], colptr[dst + 1]):
      src = rindices[offset]
      ndata[offset] = data[offset] * nrms[src]
  
  return ndata


@njit()
def _pr_numba(n_nodes, n_edges, colptr, rindices, ndata, alpha, max_iter, tol):
    """ numba graph-centric pagerank """
    
    # --
    # initialization
    
    x = np.ones(n_nodes) / n_nodes
    p = np.ones(n_nodes) / n_nodes
    
    # --
    # pagerank iterations
    
    for it in range(max_iter):
        xlast = x
        x     = (1.0 - alpha) * p
        
        for dst in range(n_nodes):
          for offset in range(colptr[dst], colptr[dst + 1]):
            src  = rindices[offset]
            nval = ndata[offset]
            x[dst] += alpha * xlast[src] * nval
        
        err = np.abs(x - xlast).max()
        if err < tol:
            return x
      
    return x


def pr_numba(adj, alpha=0.85, max_iter=100, tol=1e-6):
  """
    wrapper for numba function
    numba doesn't play nice w/ scipy.sparse matrices, so we need a function to unwrap them into arrays
  """
  n_nodes = adj.shape[0]
  n_edges = adj.nnz
  ndata   = _row_normalize(n_nodes, n_edges, adj.indptr, adj.indices, adj.data)
  return _pr_numba(n_nodes, n_edges, adj.indptr, adj.indices, ndata, alpha=alpha, max_iter=max_iter, tol=tol)
```

Note that we read the data as compressed sparse column (`csc`) instead of compressed sparse row (`csr`).  `csc` gives better memory locality inside of the pagerank iterations.

Now we can run a benchmark on the same dataset as above:
```python
{
  'read'     : 5319, 
  'convert'  : 55972, 
  'python'   : 440489, 
  'scipy'    : 796, 
  'numba_1t' : 492,
}
```

So we get about a 1.6x speedup with our hand-written `numba` implementation vs `networkx`'s scipy implementation.

`numba` also allows us to parallelize our code quite easily:
  - change `@njit()` decorators to `@njit(parallel=True)`
  - replace `for dst in range(n_nodes):` with `for dst in prange(n_nodes):`

Making these changes and running on 20 threads gives us another performance boost:
```python
{
  'read'       : 5319, 
  'convert'    : 55972, 
  'python'     : 440489, 
  'scipy'      : 796, 
  'numba_1t'   : 492,
  'numba_20t'  : 151,
}
```

That's another 3.25x speedup w/ a couple of small changes.  Note that this is nowhere close to perfect scaling (20x speedup from 20 threads), but it's still quick and easy. (Why aren't we getting perfect scaling? Load imbalance, probably -- different `dst` nodes have different numbers of neighbors -- but we won't dig any deeper on this yet.)

`numba` is a super useful tool, since it gives substantial performance improvements over native Python code, while still being easily integrated into a larger Python codebase.  It also allows you to write code in a style that would otherwise be horribly slow in Python -- in this case, using standard numpy/scipy gets you pretty good performance, but sometimes it's hard to write vectorized code that takes advantage of these libraries lower-level optimizations.

I'd love to see `networkx` start including `numba`-fied versions of more algorithms.

## Better CPU Implementation w/ C++

I won't go into the details, but if you re-write the `numba` version above using C++ and OpenMP, you can improve performance (still using the CPU) even further.  I'll add those numbers to the benchmarks for reference, and code is available in the accompanying [GitHub repo](NEED LINK):
```python
{
  'read'      : 5319, 
  'convert'   : 55972, 
  'python'    : 440489, 
  'scipy'     : 796, 
  'numba_1t'  : 492,
  'numba_20t' : 151,
  'cpp_1t'    : 337,
  'cpp_20t'   : 91
}
```

We're going to move onto GPU implementations using [gunrock/essentials] now, but it's worth emphasizing the point that a careful implementation w/ appropriate tools on the CPU reduces the runtime by > 5000x vs. a naive Python implementation and by > ~ 10x over a less naive numpy/scipy implementation.  I've found that -- in these kinds of graph-related workloads w/ lots of loopy code and lots of inherent parallelism -- speedups of this magnitude are common, and not particularly difficult to obtain once you get over the initial hump of writing numba, writing/compiling C/C++, etc.  Also note that using `pybind11` to allow the C++ implementation to be called from Python isn't particularly complex either (.. again, once you get over the initial hump).

Also note that my C++ version is not "highly optimized", and it's virtually guaranteed that you could get these runtimes down further with more effort.  However, the goal of this document is to argue that it's relatively easy to implement these kinds of algorithms in `gunrock/essentials` for a large performance improvement.

## gunrock/essentials

[`essentials`](link) is a from-scratch re-implementation of the high-performance [`gunrock`](link) library, targeting:
  - simplified application programming
  - simplified library modification

It's written and maintaing by John Owen's group at UC-Davis ECE -- @neoblizz is the primary library author.

If you're not familiar with `essentials`, you may want to look at the  [Getting Started Guide](https://github.com/bkj/essentials_guides/tree/main/0_getting_started), which walks through the structure of an `essentials` application.  However, hopefully the information below can give you a sense of what writing an `essentials` application looks like (not _too_ difficult, even without prior GPU experience), and the kinds of performance improvements you can expect (big!).

