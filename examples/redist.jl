using MPI, MPIArrays

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nb_procs = MPI.Comm_size(comm)
@assert nb_procs == 4

N = 4 # size of the matrix

# Sequentially display the partitons
function display_partitions(a::MPIArray, message)
  if rank != 0
    return
  end

  println("---- $message ----")

  for r in 0:nb_procs-1
    println("local matrix for rank $r:")
    display(getblock(a[localindices(a,r)...]))
    println()
  end
end

# Initial tiled distribution:
A = MPIArray{Int}(comm, (2, 2), N, N)
forlocalpart!(lp -> fill!(lp, rank), A)
sync(A)
display_partitions(A, "Initial distribution")

# Rows only distro
redistribute!(A, 4, 1)
display_partitions(A, "Rows")

#  non-uniform
redistribute!(A, [N], [1,0,0,3])
display_partitions(A, "Non-uniform")

redistribute!(A)
display_partitions(A, "Restored default")

if rank == 0
  println("-------Complete array -------")
  display(A)
  println()
end

# Clean up
free(A)
MPI.Finalize()
