using MPI
using MPIArrays
using Base.Test
using Compat

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nb_procs = MPI.Comm_size(comm)

if rank == 0
@testset "Partitioning" begin

const matpart = MPIArrays.ContinuousPartitioning([2,5,3], [2,3])
@test matpart[1,1] == (1:2,1:2)
@test matpart[3,2] == (8:10,3:5)
@test MPIArrays.partition_sizes(matpart) == ([2,5,3], [2,3])

const vecpart = MPIArrays.ContinuousPartitioning([2,5,3])
@test vecpart[1] == (1:2,)
@test vecpart[2] == (3:7,)
@test vecpart[3] == (8:10,)
@test MPIArrays.partition_sizes(vecpart) == ([2,5,3],)

end

end

@testset "ArrayInterface" begin

function do_test(partitions::Vararg{Int,N}) where N
    sizes = (partitions .* 4) .+ (partitions .% (1:N))
    serial_A = reshape(1:prod(sizes), sizes...)
    A = MPIArray{Int}(comm, partitions, sizes...)
    forlocalpart!(lp -> lp .= serial_A[localindices(A, rank)...], A)
    @test all(A .== serial_A)
    sync(A)
    if rank == 1
        A .= serial_A
    end
    sync(A)
    @test all(A .== serial_A)
    return A
end

vec = do_test(nb_procs)
@test size(vec) == (nb_procs*4,)

if nb_procs % 2 == 0
    p1 = nb_procs ÷ 2
    p2 = nb_procs ÷ p1
else
    p1 = nb_procs
    p2 = 1
end
@test p1*p2 == nb_procs
mat = do_test(p1, p2)

ten = MPIArray{Float64}(comm, (1,1,nb_procs), 5,6,7)
sync(ten)
@test size(ten.partitioning) == (1,1,nb_procs)

forlocalpart!(lp -> lp .= rank+1, vec)
forlocalpart!(lp -> lp .= rank+1, mat)
MPI.Win_sync(mat.win)
MPI.Win_sync(vec.win)

sync(mat)

ab1 = MPIArrays.Block(mat, 3:size(mat,1), 3:5)
ab2 = MPIArrays.Block(mat, 1:4, 1:3)
ab3 = MPIArrays.Block(mat, 1:2, 2:5)
vb = MPIArrays.Block(vec, 3:length(vec))

if rank == 0
    println("vector:")
    display(vec)
    println()
    println("matrix:")
    display(mat)
    println()
end

sync(mat)

mat_redist = redistribute(mat, 1, nb_procs)

sync(mat_redist)

if rank == 0
    @test all(mat_redist .== mat)
end

sim1 = similar(mat, Float64)
@test sim1.partitioning == mat.partitioning
@test typeof(sim1[1]) == Float64
@test size(sim1) == size(mat)
sim2 = similar(mat, (10,))
@test size(sim2) == (10,)
sim3 = similar(mat, (3,4,5))
@test size(sim3) == (3,4,5)

sync(sim3)

end

@testset "ArrayBlock" begin

dims = MPIArray{Int}(comm, (nb_procs,), 2,)

if rank == 0
    dims[1] = rand(nb_procs:5*nb_procs+1)
    dims[2] = rand(nb_procs+1:5*nb_procs)
end

sync(dims)

nrows = dims[1]
ncols = dims[2]

if rank == 0
    println("Running ArrayBlock test with size $nrows×$ncols")
end

mat1 = MPIArray{Float64}(comm, (nb_procs,1), nrows, ncols)
forlocalpart!(rand!, mat1)
sync(mat1)

localblock = mat1[localindices(mat1)...]
localmat = allocate(localblock)
getblock!(localmat, localblock)
forlocalpart(mat1) do A
    @test(all(A .== localmat))
end

serial_mat2 = reshape(1:nrows*ncols,nrows,ncols)
mat2 = MPIArray{Int}(comm, (1,nb_procs), nrows, ncols)
forlocalpart!(mat2) do A
    A .= serial_mat2[localindices(mat2)...]
end

sync(mat2)

fullblock = mat2[1:nrows, 1:ncols]
fullmat = allocate(fullblock)
getblock!(fullmat, fullblock)
@test all(fullmat .== serial_mat2)

sync(mat2)

forlocalpart!(mat2) do A
    fill!(A,0)
end

sync(mat2)
putblock!(fullmat, fullblock)
sync(mat2)

forlocalpart!(mat2) do A
    @test all(A .== serial_mat2[localindices(mat2)...])
end

sync(mat2)
putblock!(fullmat, fullblock, +)
sync(mat2)

forlocalpart!(mat2) do A
    @test all(A .== serial_mat2[localindices(mat2)...] .* (nb_procs+1))
end

end

@testset "Filter" begin

localarray = fill(rank+1, 2*(rank+1))
vec_from_local = MPIArray(localarray, nb_procs)
sync(vec_from_local)
@test size(vec_from_local) == (sum(1:nb_procs)*2,)
@test forlocalpart(lp -> all(lp .== localarray), vec_from_local)

localarray = fill(rank+1, 5, 2*(rank+1))
mat_from_local = MPIArray(localarray, 1, nb_procs)
@test size(mat_from_local) == (5, sum(1:nb_procs)*2,)
@test forlocalpart(lp -> all(lp .== localarray), mat_from_local)

testrange = collect(1:100)
mpirange = MPIArray{Int}(length(testrange))
if rank == 0
    putblock!(testrange,mpirange[:])
end
sync(mpirange)

filtered_range = filter(isodd, mpirange)
@test length(filtered_range) == length(testrange) ÷ 2
@test all(filtered_range .== filter(isodd, testrange))

localstart_before_filter = localindices(mpirange)[1][1]

sync(mpirange)

filter!(x -> x <= 25, mpirange)

if localstart_before_filter > 25
    @test forlocalpart(lp -> length(lp) == 0, mpirange)
end

sync(mpirange)

redist_range = redistribute(mpirange)
@test forlocalpart(lp -> length(lp) >= 25 ÷ nb_procs, redist_range)

end

MPI.Finalize()