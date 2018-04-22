module MPIArrays

export MPIArray, localindices, getblock, getblock!, putblock!, allocate, forlocalpart, forlocalpart!, free, redistribute, redistribute!, sync

using MPI
using Compat

"""
Store the distribution of the array indices over the different partitions.
This class forces a continuous, ordered distribution without overlap

    ContinuousPartitioning(partition_sizes...)

Construct a distribution using the number of elements per partition in each direction, e.g:

```
p = ContinuousPartitioning([2,5,3], [2,3])
```

will construct a distribution containing 6 partitions of 2, 5 and 3 rows and 2 and 3 columns.

"""
struct ContinuousPartitioning{N} <: AbstractArray{Int,N}
    ranks::LinearIndices{N,NTuple{N,Base.OneTo{Int}}}
    index_starts::NTuple{N,Vector{Int}}
    index_ends::NTuple{N,Vector{Int}}

    function ContinuousPartitioning(partition_sizes::Vararg{Any,N}) where {N}
        index_starts = Vector{Int}.(length.(partition_sizes))
        index_ends = Vector{Int}.(length.(partition_sizes))
        for (idxstart,idxend,nb_elems_dist) in zip(index_starts,index_ends,partition_sizes)
            currentstart = 1
            currentend = 0
            for i in eachindex(idxstart)
                currentend += nb_elems_dist[i]
                idxstart[i] = currentstart
                idxend[i] = currentend
                currentstart += nb_elems_dist[i]
            end
        end
        ranks = LinearIndices(length.(partition_sizes))
        return new{N}(ranks, index_starts, index_ends)
    end
end

Base.IndexStyle(::Type{ContinuousPartitioning{N}}) where {N} = IndexCartesian()
Base.size(p::ContinuousPartitioning) = length.(p.index_starts)
@inline function Base.getindex(p::ContinuousPartitioning{N}, I::Vararg{Int, N}) where {N}
    return UnitRange.(getindex.(p.index_starts,I), getindex.(p.index_ends,I))
end

partition_sizes(p::ContinuousPartitioning) = (p.index_ends .- p.index_starts .+ 1)

"""
  (private method)

Get the rank and local 0-based index
"""
function local_index(p::ContinuousPartitioning, I::NTuple{N,Int}) where {N}
    proc_indices = searchsortedfirst.(p.index_ends, I)
    return (p.ranks[proc_indices...]-1, LinearIndices(p[proc_indices...])[I...]-1)
end

# Evenly distribute nb_elems over parts partitions
function distribute(nb_elems, parts)
    local_len = nb_elems ÷ parts
    remainder = nb_elems % parts
    return [p <= remainder ? local_len+1 : local_len for p in 1:parts]
end

mutable struct MPIArray{T,N} <: AbstractArray{T,N}
    sizes::NTuple{N,Int}
    localarray::Array{T,N}
    partitioning::ContinuousPartitioning{N}
    comm::MPI.Comm
    win::MPI.Win
    myrank::Int

    function MPIArray{T}(comm::MPI.Comm, partition_sizes::Vararg{AbstractVector{<:Integer},N}) where {T,N}
        nb_procs = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        partitioning = ContinuousPartitioning(partition_sizes...)

        localarray = Array{T}(length.(partitioning[rank+1]))
        win = MPI.Win()
        MPI.Win_create(localarray, MPI.INFO_NULL, comm, win)
        sizes = sum.(partition_sizes)
        return new{T,N}(sizes, localarray, partitioning, comm, win, rank)
    end

    MPIArray{T}(sizes::Vararg{<:Integer,N}) where {T,N} = MPIArray{T}(MPI.COMM_WORLD, (MPI.Comm_size(MPI.COMM_WORLD), ones(Int,N-1)...), sizes...)
    MPIArray{T}(comm::MPI.Comm, partitions::NTuple{N,<:Integer}, sizes::Vararg{<:Integer,N}) where {T,N} = MPIArray{T}(comm, distribute.(sizes, partitions)...)

    function MPIArray(comm::MPI.Comm, localarray::Array{T,N}, nb_partitions::Vararg{<:Integer,N}) where {T,N}
        nb_procs = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        
        partition_size_array = reshape(MPI.Allgather(size(localarray), comm), Int.(nb_partitions)...)

        partition_sizes = ntuple(N) do dim
            idx = ntuple(i -> i == dim ? Colon() : 1,N)
            return getindex.(partition_size_array[idx...],dim)
        end

        win = MPI.Win()
        MPI.Win_create(localarray, MPI.INFO_NULL, comm, win)
        result = new{T,N}(sum.(partition_sizes), localarray, ContinuousPartitioning(partition_sizes...), comm, win, rank)
        return result
    end

    MPIArray(localarray::Array{T,N}, nb_partitions::Vararg{<:Integer,N}) where {T,N} = MPIArray(MPI.COMM_WORLD, localarray, nb_partitions...)
end


Base.IndexStyle(::Type{MPIArray{T,N}}) where {T,N} = IndexCartesian()

Base.size(a::MPIArray) = a.sizes

# Individual element access. WARNING: this is slow
function Base.getindex(a::MPIArray{T,N}, I::Vararg{Int, N}) where {T,N}
    (target_rank, locind) = local_index(a.partitioning,I)
    
    result = Ref{T}()
    MPI.Win_lock(MPI.LOCK_SHARED, target_rank, 0, a.win)
    if target_rank == a.myrank
        result[] = a.localarray[locind+1]
    else
        MPI.Get(result, 1, target_rank, locind, a.win)
    end
    MPI.Win_unlock(target_rank, a.win)
    return result[]
end

# Individual element setting. WARNING: this is slow
function Base.setindex!(a::MPIArray{T,N}, v, I::Vararg{Int, N}) where {T,N}
    (target_rank, locind) = local_index(a.partitioning,I)
    
    MPI.Win_lock(MPI.LOCK_EXCLUSIVE, target_rank, 0, a.win)
    if target_rank == a.myrank
        a.localarray[locind+1] = v
    else
        result = Ref{T}(v)
        MPI.Put(result, 1, target_rank, locind, a.win)
    end
    MPI.Win_unlock(target_rank, a.win)
end

"""
    sync(a::MPIArray)

Collective call, making sure all operations modifying any part of the array are finished when it completes
"""
sync(a::MPIArray, ::Vararg{MPIArray, N}) where N = MPI.Barrier(a.comm)

function Base.similar(a::MPIArray, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
    old_partition_sizes = partition_sizes(a.partitioning)
    old_dims = size(a)
    new_partition_sizes = Vector{Int}[]
    remaining_nb_partitons = prod(length.(old_partition_sizes))
    for i in eachindex(dims)
        if i <= length(old_dims)
            if dims[i] == old_dims[i]
                push!(new_partition_sizes, old_partition_sizes[i])
            else
                push!(new_partition_sizes, distribute(dims[i], length(old_partition_sizes[i])))
            end
        elseif remaining_nb_partitons != 1
            push!(new_partition_sizes, distribute(dims[i], remaining_nb_partitons))
        else
            push!(new_partition_sizes, [dims[i]])
        end
        @assert remaining_nb_partitons % length(last(new_partition_sizes)) == 0
        remaining_nb_partitons ÷= length(last(new_partition_sizes))
    end
    if remaining_nb_partitons > 1
        remaining_nb_partitons *= length(last(new_partition_sizes))
        new_partition_sizes[end] = distribute(dims[end], remaining_nb_partitons)
        remaining_nb_partitons ÷= length(last(new_partition_sizes))
    end
    @assert remaining_nb_partitons == 1
    return MPIArray{T}(a.comm, new_partition_sizes...)
end

function Base.filter(f,a::MPIArray)
    error("filter is only supported on 1D MPIArrays")
end

function Base.filter(f,a::MPIArray{T,1}) where T
    return MPIArray(forlocalpart(v -> filter(f,v), a), length(a.partitioning))
end

function Base.filter!(f,a::MPIArray)
    error("filter is only supported on 1D MPIArrays")
end

function copy_into!(dest::MPIArray{T,N}, src::MPIArray{T,N}) where {T,N}
    free(dest)
    dest.sizes = src.sizes
    dest.localarray = src.localarray
    dest.partitioning = src.partitioning
    dest.comm = src.comm
    dest.win = src.win
    dest.myrank = src.myrank
    return dest
end

Base.filter!(f,a::MPIArray{T,1}) where T = copy_into!(a, filter(f,a))

function redistribute(a::MPIArray{T,N}, partition_sizes::Vararg{Any,N}) where {T,N}
    rank = MPI.Comm_rank(a.comm)
    @assert prod(length.(partition_sizes)) == MPI.Comm_size(a.comm)
    partitioning = ContinuousPartitioning(partition_sizes...)
    localarray = getblock(a[partitioning[rank+1]...])
    return MPIArray(a.comm, localarray, length.(partition_sizes)...)
end

function redistribute(a::MPIArray{T,N}, nb_parts::Vararg{Int,N}) where {T,N}
    return redistribute(a, distribute.(size(a), nb_parts)...)
end

function redistribute(a::MPIArray)
    return redistribute(a, size(a.partitioning)...)
end

redistribute!(a::MPIArray{T,N}, partition_sizes::Vararg{Any,N})  where {T,N} = copy_into!(a, redistribute(a, partition_sizes...))
redistribute!(a::MPIArray) = redistribute!(a, size(a.partitioning)...)

function Base.A_mul_B!(y::MPIArray{T,1}, A::MPIArray{T,2}, b::MPIArray{T,1}) where {T}
    forlocalpart!(y) do ly
        fill!(ly,zero(T))
    end
    sync(y)
    (rowrng,colrng) = localindices(A)
    my_b = getblock(b[colrng])
    yblock = y[rowrng]
    my_y = allocate(yblock)
    forlocalpart(A) do my_A
        Base.A_mul_B!(my_y,my_A,my_b)
    end
    putblock!(my_y,yblock,+)
    sync(y)
    return y
end

"""
    localindices(a::MPIArray, rank::Integer)

Get the local index range (expressed in terms of global indices) of the given rank
"""
@inline localindices(a::MPIArray, rank::Integer=a.myrank) = a.partitioning[rank+1]

"""
    forlocalpart(f, a::MPIArray)

Execute the function f on the part of a owned by the current rank. It is assumed f does not modify the local part.
"""
function forlocalpart(f, a::MPIArray)
    MPI.Win_lock(MPI.LOCK_SHARED, a.myrank, 0, a.win)
    result = f(a.localarray)
    MPI.Win_unlock(a.myrank, a.win)
    return result
end

"""
    forlocalpart(f, a::MPIArray)

Execute the function f on the part of a owned by the current rank. The local part may be modified by f.
"""
function forlocalpart!(f, a::MPIArray)
    MPI.Win_lock(MPI.LOCK_EXCLUSIVE, a.myrank, 0, a.win)
    result = f(a.localarray)
    MPI.Win_unlock(a.myrank, a.win)
    return result
end

function linear_ranges(indexblock)
    cr = CartesianIndices(indices(indexblock)[2:end])
    result = Vector{UnitRange{Int}}(length(cr))

    for (i,carti) in enumerate(cr)
        linrange = indexblock[:,carti]
        result[i] = linrange[1]:linrange[end]
    end
    return result
end

struct Block{T,N}
    array::MPIArray{T,N}
    ranges::NTuple{N,UnitRange{Int}}
    targetrankindices::CartesianIndices{N,NTuple{N,UnitRange{Int}}}

    function Block(a::MPIArray{T,N}, ranges::Vararg{AbstractRange{Int},N}) where {T,N}
        start_indices =  searchsortedfirst.(a.partitioning.index_ends, first.(ranges))
        end_indices = searchsortedfirst.(a.partitioning.index_ends, last.(ranges))
        targetrankindices = CartesianIndices(UnitRange.(start_indices, end_indices))

        return new{T,N}(a, UnitRange.(ranges), targetrankindices)
    end
end

Base.getindex(a::MPIArray{T,N}, I::Vararg{UnitRange{Int},N}) where {T,N} = Block(a,I...)
Base.getindex(a::MPIArray{T,N}, I::Vararg{Colon,N}) where {T,N} = Block(a,indices(a)...)


"""
Allocate a local array with the size to fit the block
"""
allocate(b::Block{T,N}) where {T,N} = Array{T}(length.(b.ranges))

function blockloop(a::AbstractArray{T,N}, b::Block{T,N}, locktype::MPI.LockType, localfun, mpifun) where {T,N}
    for rankindex in b.targetrankindices
        r = b.array.partitioning.ranks[rankindex] - 1
        localinds = localindices(b.array,r)
        globalrange = b.ranges .∩ localinds
        a_range = globalrange .- first.(b.ranges) .+ 1
        b_range = globalrange .- first.(localinds) .+ 1
        MPI.Win_lock(locktype, r, 0, b.array.win)
        if r == b.array.myrank
            localfun(a, a_range, b.array.localarray, b_range)
        else
            alin = LinearIndices(a)
            blin = LinearIndices(length.(localinds))
            for (ai,bi) in zip(CartesianIndices(a_range[2:end]),CartesianIndices(b_range[2:end]))
                a_linear_range = alin[a_range[1][1],ai]:alin[a_range[1][end],ai]
                b_begin = blin[b_range[1][1],bi] - 1
                range_len = length(a_linear_range)
                @assert b_begin + range_len <= prod(length.(localinds))
                mpifun(pointer(a,first(a_linear_range)), range_len, r, b_begin, b.array.win)
            end
        end
        MPI.Win_unlock(r, b.array.win)
    end
end

"""
Get the (possibly remotely stored) elements of the block and store them in a
"""
function getblock!(a::AbstractArray{T,N}, b::Block{T,N}) where {T,N}
    localfun = function(local_a, a_range, local_b, b_range)
        local_a[a_range...] .= local_b[b_range...]
    end
    blockloop(a, b, MPI.LOCK_SHARED, localfun, MPI.Get)
end

"""
Allocate a matrix to store the data referred to by block and copy all elements from the global array to it
"""
function getblock(b::Block{T,N}) where {T,N}
    a = allocate(b)
    getblock!(a,b)
    return a
end

@inline _no_op(a,b) = b
_mpi_put_with_op(origin, count, rank, disp, win, op) = throw(ErrorException("Unsupported op $op"))
_mpi_put_with_op(origin, count, rank, disp, win, op::typeof(_no_op)) = MPI.Put(origin, count, rank, disp, win)
_mpi_put_with_op(origin, count, rank, disp, win, op::typeof(+)) = MPI.Accumulate(origin, count, rank, disp, MPI.SUM, win)

"""
Set the (possibly remotely stored) elements of the block and store them in a
"""
function putblock!(a::AbstractArray{T,N}, b::Block{T,N}, op::Function=_no_op) where {T,N}
    localfun = function(local_a, a_range, local_b, b_range)
        local_b[b_range...] .= op.(local_b[b_range...], local_a[a_range...])
    end
    mpifun = function(origin, count, target_rank, target_disp, win)
        _mpi_put_with_op(origin, count, target_rank, target_disp, win, op)
    end
    blockloop(a, b, MPI.LOCK_EXCLUSIVE, localfun, mpifun)
end

function free(a::MPIArray{T,N}) where {T,N}
    sync(a)
    MPI.Win_free(a.win)
end

end # module
