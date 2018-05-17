using MPI, MPIArrays

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nb_procs = MPI.Comm_size(comm)

N = 20 # size of the matrix

# Initial tiled distribution:
A = MPIArray{Float64}(comm, (nb_procs, 1), N, N)
forlocalpart!(lp -> fill!(lp, rank), A)
sync(A)

if rank == 0
  using Plots
  heatmap(getblock(A[:,:]))
  savefig("plot.png")
end

# Clean up
free(A)
MPI.Finalize()
