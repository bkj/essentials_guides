#include <gunrock/applications/application.hxx>

// Include your application code -- we'll comment this out now so we can compile a test.
#include <gunrock/applications/my_sssp.hxx>

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

  // --
  // Params and memory allocation
  
  vertex_t single_source = 0;
  
  vertex_t n_vertices    = G.get_number_of_vertices();
  thrust::device_vector<weight_t> distances(n_vertices);

  // --
  // GPU Run

  float gpu_elapsed = gunrock::my_sssp::run(G, single_source, distances.data().get());

  // --
  // Log + Validate

  std::cout << "GPU Distances (output) = ";
  thrust::copy(distances.begin(), distances.end(), std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

// Main method, wrapping test function
int main(int argc, char** argv) {
  test_my_sssp(argc, argv);
}
