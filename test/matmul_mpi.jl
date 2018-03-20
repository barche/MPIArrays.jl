using MPI
using MPIArrays
using Base.Test

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nb_procs = MPI.Comm_size(comm)

do_serial = (rank == 0)

include("matmul_setup.jl")

function timed_mul(A::MPIArray,x::MPIArray)
    for i in 1:3
        if rank == 0
            @time begin
                b = A*x
                MPI.Barrier(A.comm)
            end
        else
            b = A*x
            MPI.Barrier(A.comm)
        end
        free(b)
    end
end

@testset "MatMul" begin

const A1 = MPIArray{Float64}(comm, (nb_procs,1), N, N)
const A2 = MPIArray{Float64}(comm, (1,nb_procs), N, N)
const x = MPIArray{Float64}(comm, (nb_procs,), N)

if rank == 0
    putblock!(As,A1[:,:])
    putblock!(As,A2[:,:])
    putblock!(xs,x[:])
end

const pres1 = A1*x
const pres2 = A2*x

if rank == 0
    @test all(getblock(pres1[:]) .≈ ref)
    @test all(getblock(pres2[:]) .≈ ref)
end

MPI.Barrier(comm)

rank == 0 && println("Parallel, rows distributed:")
timed_mul(A1,x)
rank == 0 && println("Parallel, columns distributed:")
timed_mul(A2,x)

MPI.Barrier(comm)

end

MPI.Finalize()