using MPI, MPIArrays

MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
N = 30 # size of the matrix

# Create an uninitialized matrix and vector
x = MPIArray{Float64}(N)
A = MPIArray{Float64}(N,N)

# Set random values
forlocalpart!(rand!,x)
forlocalpart!(rand!,A)

# Make sure every process finished initializing the coefficients
sync(A, x)

b = A*x

# This will print on the first process, using slow element-by-element communication, but that's OK to print to screen
rank == 0 && println("Matvec result: $b")

# Clean up
free(A)
free(x)
MPI.Finalize()
