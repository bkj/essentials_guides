# `Essentials` Application Programming Essentials

This document is intended as a reference for someone trying to implement their first `gunrock/essentials` application.

As an example, I'll walk you through re-implementing single-source shortest paths (`sssp`).  This application already exists in `essentials`, but we'll use a different name (`my_sssp`) so you can see all of the steps.

## Application Overview

Before writing any `essentials` code, let's walk through the single-source shortest paths algorithm that we'll be implementing.  We'll implement a fully functional algorithm in Python, so you can test and modify an actual piece of code.

Note that this SSSP algorithm looks quite a bit different from a standard Dijkstra's algorithm.  This kind of algorithmic re-formulation is often necessary to expose a lot of parallelism in the algorithm and to use the highly-optimized operators in `essentials`.  (If you're familiar w/ `gunrock` programming, this kind of thing should be familiar.)

As we go through the `essentials` implementation, refer back to this implementation to see the one-to-one correspondence between this simple (and very slow...) Python implementation and the fast `essentials` code.

Details are given inline below.

```python
import numpy as np
from scipy.io import mmread

def my_sssp(csr, single_source, distances):
  # initialize distance array
  distances[:]             = np.inf
  distances[single_source] = 0
  
  # initialize internal datastructures
  n_vertices = csr.shape[0]
  visited    = np.zeros(n_vertices) - 1
  
  # intialize the frontier
  frontier_in  = [single_source]
  frontier_out = []
  
  iteration = 0
  while len(frontier_in) > 0:
    
    # --
    # advance step
    
    # For each vertex `source` in the frontier:
    #  For each `neighbor` of `source` in the graph:
    #   Check whether path to `neighbor` through `source` is shorter than current distance to `neighbor`
    #   If so, update distance and add `neighbor` to frontier of active vertices
    
    for source in frontier_in:
      
      neighbors = csr.indices[csr.indptr[source]:csr.indptr[source + 1]]
      weights   = csr.data[csr.indptr[source]:csr.indptr[source + 1]]
      
      for neighbor, weight in zip(neighbors, weights):
        
        new_dist = distances[source] + weight
        old_dist = distances[neighbor]
        distances[neighbor] = min(distances[neighbor], new_dist)
        
        if new_dist < old_dist:
          frontier_out.append(neighbor)
    
    frontier_in  = frontier_out
    frontier_out = []
    
    # --
    # filter step
    
    # Remove vertices w/ 0 outdegree from the frontier; and
    # De-duplicate the frontier; and
    # Record step at which the shortest path to `vertex` was found
    
    for vertex in frontier_in:
      
      n_neighbors = csr.indptr[source] - csr.indptr[source + 1]
      if n_neighbors == 0: continue
      
      if visited[vertex] == iteration: continue
      visited[vertex] = iteration
      
      frontier_out.append(vertex)
    
    frontier_in  = frontier_out
    frontier_out = []
    
    iteration += 1
  

# --
# Run w/ small test dataset

csr           = mmread('../datasets/chesapeake.mtx').tocsr()
n_vertices    = csr.shape[0]
single_source = 0
distances     = np.zeros(n_vertices)
my_sssp(csr, single_source, distances)
print(distances)
```


## Directory Structure

To implement `my_sssp`, you'll need to modify 1 existing file:
```
examples/CMakeLists.txt
```

and create 3 new files:
```
examples/my_sssp/CMakeLists.txt
examples/my_sssp/my_sssp.cu
include/gunrock/applications/my_sssp.hxx
```

## 0) Before getting started ...

Depending on your GPU, you may also want to edit the `CUDA_ARCHITECTURES` variable in `$PROJECT_ROOT/CMakeLists.txt`.  

It is set to `61` by default, but newer GPUs may require a different value (eg, `70` for `V100`).


## 1) Modify `examples/CMakeLists.txt`

To tell the build system about `my_sssp`, add the following line to `examples/CMakeLists.txt`:
```
add_subdirectory(my_sssp)
```

## 2) Create `examples/my_sssp/CMakeLists.txt`

```bash
cd $PROJECT_ROOT

# create directory
mkdir examples/my_sssp

# copy boilerplate CMakeLists.txt to `examples/my_sssp`
cp examples/bfs/CMakeLists.txt examples/my_sssp/CMakeLists.txt

# change APPLICATION_NAME from `bfs` to `my_sssp`
sed -i "s/set(APPLICATION_NAME bfs)/set(APPLICATION_NAME my_sssp)/" examples/my_sssp/CMakeLists.txt
```

## 3) Create `examples/my_sssp/my_sssp.cu`

First we'll write a short command-line test driver for the application.  This might seem a little backwards -- we're writing the function that _calls_ the application before we write the application itself.  However, I think this is helpful, but it allows you to test compilation and file IO before you get to application development.  

### a) Write command line interface and IO code.  

This is boilerplate code that'll be the same for most applications.

Read the comments inline below for details.

```c++
#include <gunrock/applications/application.hxx>

// Include the application code -- we'll comment this out now so we can compile a test quickly.
// #include <gunrock/applications/my_sssp.hxx>

using namespace gunrock;
using namespace memory;

void test_my_sssp(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types
  // Specify the types that will be used for
  // - vertex ids (vertex_t)
  // - edge offsets (edge_t)
  // - edge weights (weight_t)
  
  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;

  // --
  // IO
  
  // Filename to be read
  std::string filename = argument_array[1];

  // Load the matrix-market dataset into csr format.
  // See `format` to see other supported formats.
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  using csr_t = format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;
  csr.from_coo(mm.load(filename));

  // --
  // Build graph

  // Convert the dataset you loaded into an `essentials` graph.
  // `memory_space_t::device` -> the graph will be created on the GPU.
  // `graph::view_t::csr`     -> your input data is in `csr` format.
  //
  // Note that `graph::build::from_csr` expects pointers, but the `csr` data arrays
  // are `thrust` vectors, so we need to unwrap them w/ `.data().get()`.
  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,
      csr.number_of_columns,
      csr.number_of_nonzeros,
      csr.row_offsets.data().get(),
      csr.column_indices.data().get(),
      csr.nonzero_values.data().get() 
  );

  std::cout << "G.get_number_of_vertices() : " << G.get_number_of_vertices() << std::endl;
  std::cout << "G.get_number_of_edges()    : " << G.get_number_of_edges()    << std::endl;
}

// Main method, wrapping test function
int main(int argc, char** argv) {
  test_my_sssp(argc, argv);
}
```

At this point, you can try compiling your code:
```bash
cd $PROJECT_ROOT
mkdir build 
cd build
cmake ..
make my_sssp -j12
```

If compilation completes without errors, you can test your program:
```
cd $PROJECT_ROOT/build
./bin/my_sssp ../datasets/chesapeake.mtx
```
which should print:
```
G.get_number_of_vertices() : 39
G.get_number_of_edges()    : 340
```

Hopefully this works, and we can now move on to writing code specific to our single-source shortest paths application.

### b) Initialization + calling application code

Single-source shortest paths has 1 parameter, the source of the search.  We'll call this `single_source`.

Our implementation will output one array, the distance between `single_source` and the other nodes in the graph.  We'll call this array `distances`.

Read the comments inline below for details.

```c++
  // ...
  
  std::cout << "G.get_number_of_vertices() : " << G.get_number_of_vertices() << std::endl;
  std::cout << "G.get_number_of_edges()    : " << G.get_number_of_edges()    << std::endl;

  // --
  // Params and memory allocation
  
  // Set single_source
  // You'd probably actually want to pass this as a command-line parameter, but let's
  // hard-code it for now.
  vertex_t single_source = 0;
  
  // Initialize a `thrust::device_vector` of length `n_vertices` for distances
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> distances(n_vertices);

  // --
  // Run

  // Call the gunrock function to run `my_sssp`
  // Note that this code doesn't exist yet, so this will break the compiler, 
  // but we'll be creating it in the next step
  float gpu_elapsed = gunrock::my_sssp::run(G, single_source, distances.data().get());

  // --
  // Log + Validate

  // Use a fancy thrust function to print the results to the command line
  // Note, if your graph is big you might not want to print this whole thing
  std::cout << "GPU Distances (output) = ";
  thrust::copy(distances.begin(), distances.end(), std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;

  // Print runtime returned by `gunrock::my_sssp::run`
  // This will just be the GPU runtime of the "region of interest", and will ignore any 
  // setup/teardown code.
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_my_sssp(argc, argv);
}
```

Finally, go back to the top of the file and uncomment
```c++
#include <gunrock/applications/my_sssp.hxx>
```

so that this script can see the implementation you'll be writing in the next step.

This is the entirity of `examples/my_sssp.cu`.  As noted above, this won't compile, because `gunrock/applications/my_sssp.hxx` doesn't exist yet ... so let's move on to implementing that.

For reference, tour entire `examples/my_sssp.cu` should look like the code [here](https://github.com/bkj/essentials_guides/blob/main/0_getting_started/examples/my_sssp/my_sssp.cu).

## 4) Create `include/gunrock/applications/my_sssp.hxx`

`essentials` applications are written using four structs -- `param`, `result`, `problem` and `enactor` -- and a wrapper function -- `run`.  We'll walk through each of these below.

### a) `param` struct

The `param` struct holds all of the relevant user-defined parameters for the application.  In this case, the only parameter is `single_source`, so `param` is quite simple:
```c++
template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};
```

Note that in general it's good to keep names as consistent as possible between calling function, arguments and struct attribute.  `essentials` uses the convention that the function argument is decorated with `_` prefix, and the struct attribute uses an undecorated name.

### b) `result` struct

The `result` struct holds all of the data structures that you'll want to return to the user.  In this case, the only data structures is `distances`.  Note that we do _not_ put internal datastructures that are used by the application here -- those will go in `problem.`
```c++
template <typename vertex_t, typename weight_t>
struct result_t {
  weight_t* distances;
  result_t(weight_t* _distances) : distances(_distances) {}
};
```

### c) `problem` struct

The `problem` struct holds data structures that are used internally by the application. 

`problem` has two methods -- `init` and `reset` -- which are described inline below.

In the code blocks below, `<boilerplate>` tags indicate sections of code that are probably the same for every application. 
 
```c++
// <boilerplate>
template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(
    graph_t& G,
    param_type& _param,
    result_type& _result,
    std::shared_ptr<cuda::multi_context_t> _context
  ) : gunrock::problem_t<graph_t>(G, _context), param(_param), result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
// </boilerplate>
```

#### `init` method

`init` is a method that should be called the first time `problem` is instantiated with a dataset.  `init` should allocate memory for internal datastructures and compute any necessary internal parameters.

```c++
  // Create a datastructure that is internal to the application, and will not be returned to
  // the user.
  thrust::device_vector<vertex_t> visited;

  // `init` function, described above.  This should be called once, when `problem` gets instantiated.
  void init() {
    // Get the graph
    auto g = this->get_graph();
    
    // Get number of vertices from the graph
    auto n_vertices = g.get_number_of_vertices();   
    
    // Set the size of `visited` (`thrust` function)
    visited.resize(n_vertices);
  }

  // `reset` function, described above.  Should be called
  // - after init, when `problem` is instantiated
  // - between subsequent application runs, eg when you change the parameters
```

#### `reset` method

The `reset` method  should be called if you want to run the same application multiple times on the same dataset (eg, with different parameters). 

For example, in this case maybe you'd want to run `my_sssp` with 10 different `single_source` seeds -- in that case, you call `init` _and_ `reset` after creating `problem` the first time, and `reset` in between each subsequent run.
 
```c++
  void reset() {
    auto g = this->get_graph();
    
    auto distances  = this->result.distances;
    auto n_vertices = g.get_number_of_vertices();
    
    // fill `distances` with the max `weight_t` value
    // ... because at the beginning of `sssp`, distance to all non-source nodes should be infinity
    thrust::fill(
      thrust::device, 
      distances + 0, 
      distances + n_vertices,
      std::numeric_limits<weight_t>::max()
    );

    // Set the `single_source`'th element of distances to 0
    // ... because at the beginning of `sssp`, distance to the source node should be 0
    thrust::fill(
      thrust::device,
      distances + this->param.single_source,
      distances + this->param.single_source + 1,
      0
    );

    // Fill `visited` with -1 (`thrust` function)
    // ... because at the beginning of `sssp`, no nodes have been
    thrust::fill(thrust::device, visited.begin(), visited.end(), -1);
  }
};

```

### c) `enactor` struct

The `enactor` struct is where the computation of the application actually happens. 

The high level logic of the `essentials` enactor looks like:
```c++
float enact() {
  prepare_frontier(get_input_frontier());
  while (!is_converged(*context)) {
    loop(*context);
  }
}
```
(See `include/gunrock/framework/enactor.hxx` for the actual implementation).

This implies that we might need to implement three methods -- `prepare_frontier`, `loop` and `is_converged` -- which are described inline below.

`<boilerplate>` is defined as above.

```c++
// <boilerplate>
template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  using gunrock::enactor_t<problem_t>::enactor_t;

  using vertex_t = typename problem_t::vertex_t;
  using edge_t   = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
// </boilerplate>
```

#### `prepare_frontier` method

`prepare_frontier` initializes the frontier for the first iteration.  For single-source shortest paths, we'll just add the single source node to the frontier.  In other applications, you may add set of nodes or all nodes to the frontier.

```c++
  // How to initialize the frontier at the beginning of the application.
  // In this case, we just need to add a single node
  void prepare_frontier(frontier_t<vertex_t>* f, cuda::multi_context_t& context) override {
    // get pointer to the problem
    auto P = this->get_problem();
    
    // add `single_source` to the frontier
    f->push_back(P->param.single_source);
  }
```

#### `loop` method

The `loop` method contains the core computational logic of your application:

```c++
  // One iteration of the application
  void loop(cuda::multi_context_t& context) override {

    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    // Get parameters and datastructures
    // Note that `P->visited` is a thrust vector, so we need to unwrap again
    auto single_source = P->param.single_source;
    auto distances     = P->result.distances;
    auto visited       = P->visited.data().get();

    // Get current iteration of application
    auto iteration = this->iteration;

    // Advance operator for single-source shortest paths application
    auto shortest_path = [distances, single_source] __host__ __device__(
      vertex_t const& source,    // source of edge
      vertex_t const& neighbor,  // destination of edge
      edge_t const& edge,        // id of edge
      weight_t const& weight     // weight of edge
    ) -> bool {
      
      // Get implied distance to neighbor using a path through source
      weight_t new_dist = distances[source] + weight;
      
      // Store min(distances[neighbor], new_dist) in distances[neighbor]
      weight_t old_dist = math::atomic::min(distances + neighbor, distance_to_neighbor);

      // If the new distance is better than the previously known best_distance, add `neighbor` to 
      // the frontier
      return new_dist < old_dist;
    };

    // Execute advance operator
    // !! TODO: More documentation on these flags
    operators::advance::execute<operators::advance_type_t::vertex_to_vertex,
                                operators::advance_direction_t::forward,
                                operators::load_balance_t::block_mapped>(
        G, E, shortest_path, context);
    
    
    auto remove_completed_paths = [G, visited, iteration] __host__ __device__(
      vertex_t const& vertex
    ) -> bool {
      
      // Drop nodes w/ no out-degree, since we can't continue search from them
      if (G.get_number_of_neighbors(vertex) == 0) return false;
      
      // Uniquify the frontier
      if (visited[vertex] == iteration) return false;
      visited[vertex] = iteration;
      
      // Otherwise, keep this node in the frontier
      return true;
    };

    // Execute filter operator
    // !! TODO: More documentation on these flags
    operators::filter::execute<operators::filter_algorithm_t::predicated>(
        G, E, remove_completed_paths, context);
  }

```

#### `is_converged` method

`is_converged` check whether the algorithm has converged.  The default `is_converged` returns `true` iff the current frontier is empty.  This is the right convergence criteria for many applications including single-source shortest paths -- so we won't write our own here.  For other applications (eg, PageRank), you'd need to implement a custom convergence criterion.

The following code is not necessary for single-source shortest paths, because it's the same as the default.  However, we'll include it here for completeness.
 
```c++
  virtual bool is_converged(cuda::multi_context_t& context) {
    return active_frontier->is_empty();
  }
};
```

### d) `run` function

You've now implemented almost everything you need to actually run your code!  The final thing is a wrapper function so that you call call your application like you did in `examples/my_sssp/my_sssp.cu`.
```
float elapsed = gunrock::my_sssp::run(G, single_dource, distances.data.get());
```

This is mostly boilerplate code (though be mindful about passing template parameters correctly).

```c++
template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::weight_type* distances       // Output
) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  // instantiate `param` and `result` templates
  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t, weight_t>;

  // initialize `param` and `result` w/ the appropriate parameters / data structures
  param_type param(single_source);
  result_type result(distances);

  // <boilerplate> This code probably should be the same across all applications, 
  // unless maybe you're doing something like multi-gpu / concurrent function calls
  
  // Context for application (eg, GPU + CUDA stream it will be executed on)
  auto multi_context =
      std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0));

  // instantiate `problem` and `enactor` templates.
  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  // initialize problem; call `init` and `reset` to prepare data structures
  problem_type problem(G, param, result, multi_context);
  problem.init();
  problem.reset();

  // initialize enactor; call enactor, returning GPU elapsed time
  enactor_type enactor(&problem, multi_context);
  return enactor.enact();
  // </boilerplate>
}
```

And now you're done! 

Your entire `include/gunrock/applications/my_sssp.hxx` file should look like the code [here](https://github.com/bkj/essentials_guides/blob/main/0_getting_started/include/essentials/applications/my_sssp.hxx).  (Note the `#pragma once` and `namespace`'s at the top.)

## Run!

Now you can compile your whole application as before:
```bash
cd $PROJECT_ROOT
mkdir build 
cd build
cmake ..
make my_sssp -j12
```

If compilation succeeds without errors, you can run your code as before:
```bash
cd $PROJECT_ROOT/build
./bin/my_sssp ../datasets/chesapeake.mtx
```

which should print something like
```
G.get_number_of_vertices() : 39
G.get_number_of_edges()    : 340
GPU Distances (output) = 0 2 2 2 2 2 1 1 2 2 1 1 1 2 2 2 2 2 2 2 2 1 1 2 2 2 2 2 2 2 2 2 2 1 1 2 1 2 1 
GPU Elapsed Time : 2.28979 (ms)
```