using MPI
using MPIArrays
using Test

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nb_procs = MPI.Comm_size(comm)

do_serial = (rank == 0)

include(joinpath(@__DIR__, "matmul_setup.jl"))

function timed_mul(A::MPIArray,x::MPIArray)
    times = zeros(nsamples)
    for i in 1:length(times)
        if rank == 0
            val, times[i], bytes, gctime, memallocs = @timed begin
                b = A*x
                sync(b)
            end
        else
            b = A*x
            sync(b)
        end
        free(b)
    end
    return extrema(times[2:end])
end

@testset "MatMul" begin

A1 = MPIArray{Float64}(comm, (nb_procs,1), N, N)
A2 = MPIArray{Float64}(comm, (1,nb_procs), N, N)
x = MPIArray{Float64}(comm, (nb_procs,), N)

if rank == 0
    putblock!(As,A1[:,:])
    putblock!(As,A2[:,:])
    putblock!(xs,x[:])
end

pres1 = A1*x
pres2 = A2*x

if rank == 0
    @test all(getblock(pres1[:]) .≈ ref)
    @test all(getblock(pres2[:]) .≈ ref)
end

sync(pres2)

global (mintime, maxtime) = timed_mul(A1,x)
rank == 0 && write_timings("out-mpi.txt", 1, mintime, maxtime)
global (mintime, maxtime) = timed_mul(A2,x)
rank == 0 && write_timings("out-mpi.txt", 2, mintime, maxtime)

sync(pres2)

end

MPI.Finalize()