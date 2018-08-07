using Test

const file_dir = @__DIR__

@sync for p in workers()
  @spawnat p global file_dir = file_dir
end

@everywhere begin
  using DistributedArrays
  do_serial = myid() == 1
  const nb_procs = nprocs()
  include(joinpath(file_dir, "matmul_setup.jl"))
end

@testset "MatMul" begin

println("DArray test on $nb_procs processes")

Ad1 = distribute(As, procs=procs(), dist=(nprocs(),1))
Ad2 = distribute(As, procs=procs(), dist=(1,nprocs()))
xd = distribute(xs, procs=procs(), dist=(nprocs(),))

dres1 = Ad1*xd
dres2 = Ad2*xd
@test all(dres1 .≈ ref)
@test all(dres2 .≈ ref)

global (mintime, maxtime) = timed_mul(Ad1,xd)
write_timings("out-darray.txt", 1, mintime, maxtime)
global (mintime, maxtime) = timed_mul(Ad2,xd)
write_timings("out-darray.txt", 2, mintime, maxtime)

end