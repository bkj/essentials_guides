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

If you're not familiar with `essentials`, I'd highly recommend looking at the [Essentials Getting Started Guide](https://github.com/bkj/essentials_guides/tree/main/0_getting_started), which walks through the structure of an `essentials` application.  However, hopefully the information below can give you a sense of what writing an `essentials` application looks like (not _too_ difficult, even without prior GPU experience), and the kinds of performance improvements you can expect (substantial!).

Logically, the `essentials` implementations is (nearly) identical to the other implementations desribed in this document -- if you get confused, you can reference the Python implementations to remember all of the steps needed for `pagerank`.

`essentials` applications are implemented using four structs -- `param`, `result`, `problem` and `enactor`.  The structs for `pagerank` are shown below.  If this is your first time seeing `essentials` code, note that some of the code below is boilerplate code, which doesn't need to change across `essentials` applications.  There's a learning curve to this, but after implementing a couple of `essentials` applications it should (hopefully) be relatively straightforward.

### `param` struct

The `param` struct holds user supplied parameters.  In this case, it's the `pagerank` `alpha` parameter (eg, random jump probability) and the convergence tolerance threshold `tol`.

```c++
template <typename weight_t>
struct param_t {
  weight_t alpha;
  weight_t tol;
  param_t(weight_t _alpha, weight_t _tol) : alpha(_alpha), tol(_tol) {}
};
```

### `result` struct

The `result` struct holds the data structures that will get passed back to the user as results.  In this case, this is just the array of `pagerank` values, `x`.

```c++
template <typename weight_t>
struct result_t {
  weight_t* x;
  result_t(weight_t* _x) : x(_x) {}
};
```

### `problem` struct

The `problem` struct holds data structures that are used internally by the application, as well as the graph we're running our application on.

`problem` also has two methods -- `init` and `reset` -- which are shown below.  Refer to the [Getting Started Guide](https://github.com/bkj/essentials_guides/tree/main/0_getting_started) for more details.

```c++
template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<cuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<weight_t> xlast;     // pagerank values from previous iteration
  
  thrust::device_vector<weight_t> iweights;  // alpha * 1 / (sum of outgoing weights) -- used to determine
                                             // out of mass spread from src to dst

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    
    // Allocate memory
    xlast.resize(n_vertices);
    iweights.resize(n_vertices);
  }

  void reset() override {
    // Execution policy for a given context (using single-gpu).
    auto policy = this->context->get_context(0)->execution_policy();

    auto g = this->get_graph();

    auto alpha      = this->param.alpha;
    auto n_vertices = g.get_number_of_vertices();
    
    // Fill `x` w/ 1 / n_vertices
    thrust::fill_n(policy, this->result.x, n_vertices, 1.0 / n_vertices);

    // Fill `xlast` with 0's
    thrust::fill_n(policy, xlast.begin(), n_vertices, 0);

    // Fill `iweights` with the sum of the outgoing edge weights,
    // by applying the `get_weight` lambda function to all nodes in the graph
    auto get_weight = [=] __device__(const int& i) -> weight_t {
      weight_t val = 0;

      edge_t start = g.get_starting_edge(i);
      edge_t end   = start + g.get_number_of_neighbors(i);
      for (edge_t offset = start; offset < end; offset++) {
        val += g.get_edge_weight(offset);
      }

      return val != 0 ? alpha / val : 0;
    };

    thrust::transform(policy, thrust::counting_iterator<vertex_t>(0),
                      thrust::counting_iterator<vertex_t>(n_vertices),
                      iweights.begin(), get_weight);
  }
};
```

### `enactor` struct

The `enactor` struct is where the computation of the application actually happens.  `enactor` has three methods:
  - `prepare_frontier` -- intialize the vertex frontier for the first iteration
  - `loop` -- logic for a single iteration of the algorithm
  - `is_converged` -- check whether the algorithm has converged and should terminate

The methods are detailed inline below.

```c++
template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  using gunrock::enactor_t<problem_t>::enactor_t;

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
```

#### prepare_frontier

`prepare_frontier` initializes the frontier for the first iteration of the algorithm.  In this case, we want the frontier to hold _all_ of the nodes in the graph.

```c++
  void prepare_frontier(frontier_t<vertex_t>* f,
                        cuda::multi_context_t& context) override {
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto n_vertices = G.get_number_of_vertices();

    // Fill the frontier with a sequence of vertices from 0 -> n_vertices.
    f->sequence((vertex_t)0, n_vertices, context.get_context(0)->stream());
  }
```

#### loop

`loop` implements a single iteration of the `pagerank` algorithm.  Logically, it is almost identical to the section of code inside the Python `while True:` loop in the implementations above.

```c++
  void loop(cuda::multi_context_t& context) override {
    // Get data structures
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto n_vertices = G.get_number_of_vertices();
    auto x = P->result.x;
    auto xlast = P->xlast.data().get();
    auto iweights = P->iweights.data().get();
    auto alpha = P->param.alpha;

    auto policy = this->context->get_context(0)->execution_policy();

    // Copy p to xlast
    thrust::copy_n(policy, x, n_vertices, xlast);
    
    // Fill `x` with (1 - alpha) / n_vertices
    thrust::fill_n(policy, x, n_vertices, (1 - alpha) / n_vertices);

    auto spread_op = [p, xlast, iweights] __host__ __device__(
                         vertex_t const& src, vertex_t const& dst,
                         edge_t const& edge, weight_t const& weight) -> bool {
      
      // Compute the update to push to neighbors
      weight_t update = xlast[src] * iweights[src] * weight;
      
      // Push the update to neighbors
      // Note this needs an atomic operation because multiple `src` nodes might be writing to the same
      // `dst` node
      math::atomic::add(x + dst, update);
      return false;
    };

    // Run an optimized `essentials` operator to map `spread_op` across all the edges of the graph
    operators::advance::execute<operators::advance_type_t::vertex_to_vertex,
                                operators::advance_direction_t::forward,
                                operators::load_balance_t::merge_path>(
        G, E, spread_op, context);

    // HACK!
    // Normally, `advance` produces a new frontier of active nodes.  However, for pagerank, we
    // want all nodes to be active all of the time.  We can make this happen by (un)flipping the input
    // and output frontiers after the `advance` operator.
    // (This should be smoothed out in future `essentials` versions.)
    E->swap_frontier_buffers();
  }
```

#### `is_converged`

Determine whether the algorithm has converged by checking whether the absolute difference between the current and last iteration is below some threshold.

This is identical to the Python implementations shown above, but written using `thrust` operators that run on the GPU.

```c++
  virtual bool is_converged(cuda::multi_context_t& context) {
    if (this->iteration == 0)
      return false;

    auto P = this->get_problem();
    auto G = P->get_graph();
    auto tol = P->param.tol;

    auto n_vertices = G.get_number_of_vertices();
    auto x = P->result.x;
    auto xlast = P->xlast.data().get();

    auto abs_diff = [=] __device__(const int& i) -> weight_t {
      return abs(x[i] - xlast[i]);
    };

    auto policy = this->context->get_context(0)->execution_policy();
    
    // Map `abs_diff` across all vertices, then take the maximum
    float err = thrust::transform_reduce(
        policy, thrust::counting_iterator<vertex_t>(0),
        thrust::counting_iterator<vertex_t>(n_vertices), abs_diff,
        (weight_t)0.0, thrust::maximum<weight_t>());

    return err < tol;
  }
};
```

Clearly, the `essentials` application is quite a bit more verbose than a Python implementation.  However, the logic is essentially the same.

A fully functional `essentials` application involves implementing a bit more wrapper code plus a command line driver to run the application.  The full code for `essentials` `pagerank` is visible in these two files:
 - Full code described above: [include/gunrock/applications/pr.hxx](https://github.com/gunrock/essentials/blob/master/include/gunrock/applications/pr.hxx)
 - Command line driver: [examples/pr/pr.cu](https://github.com/gunrock/essentials/blob/master/examples/pr/pr.cu)

Finally -- now that we've implemented our `essentials` version of `pagerank`, we can benchmark and compare against the various implementations detailed above.

```python
{
  'read'       : 5319, 
  'convert'    : 55972, 
  'python'     : 440489, 
  'scipy'      : 796, 
  'numba_1t'   : 492,
  'numba_20t'  : 151,
  'cpp_1t'     : 337,
  'cpp_20t'    : 91,
  'essentials' : 8.4     # (V100 GPU on AWS p3 instance)
}
```

The `essentials` application runs `rmat18` in 8.4ms -- which is ~ 11x faster than our fastest CPU implementation `cpp_20t` and ~ 95x faster than the `pagerank` implementation you'd likely use in `networkx`.

Also, note that we are in the process of writing Python bindings for `essentials` implemented [here](https://github.com/bkj/python_essentials).  These bindings use the `pytorch` library, so the Python user has control over moving data between CPU and GPU.  This allows `essentials` applications to be integrated tightly with other Python-based GPU code, without incurring uneccessary data transfer overhead.

## Conclusion

After reading, hopefully you have a sense of tools and methods for a plausible sequence of optimizations that an application developer might use to try to write the most performance version of their application. 

In summary, key takeaways from this experiment are:
  - Careful use of python libraries (`scipy` and `numba`) can give huge performance boosts over "base" Python
  - Yet more gains can usually be had by writing multithreaded CPU code in C or C++.
  - For the highest performance, you can take advantage of the suite of optimized operators available in the `essentials` library, without particularly deep knowledge of CUDA / GPU programming.
