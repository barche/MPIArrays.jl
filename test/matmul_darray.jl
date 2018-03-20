using Base.Test

@everywhere begin
  using DistributedArrays
  do_serial = myid() == 1
  include("matmul_setup.jl")
end

@testset "MatMul" begin

println("DArray test on $(nprocs()) processes")

const Ad1 = distribute(As, procs=procs(), dist=(nprocs(),1))
const Ad2 = distribute(As, procs=procs(), dist=(1,nprocs()))
const xd = distribute(xs, procs=procs(), dist=(nprocs(),))

const dres1 = Ad1*xd
const dres2 = Ad2*xd
@test all(dres1 .≈ ref)
@test all(dres2 .≈ ref)

println("DistributedArrays, rows distributed:")
timed_mul(Ad1,xd)
println("DistributedArrays, columns distributed:")
timed_mul(Ad2,xd)

end