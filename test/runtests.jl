# Make sure this is precompiled on only one process first
using MPIArrays

julia_exe = joinpath(Sys.BINDIR, Base.julia_exename())
mpiexec = "mpiexec"

testdir = dirname(@__FILE__)

mpifiles = ["basics.jl", "matmul_mpi.jl"]
juliafiles = ["matmul_darray.jl"]
nprocs = clamp(Sys.CPU_THREADS ÷ 2, 2, 4)
ENV["OPENBLAS_NUM_THREADS"] = 1

for f in mpifiles
    cmd = `$mpiexec -n $nprocs $julia_exe --startup-file=no $(joinpath(testdir, f))`
    println("Executing $cmd")
    run(cmd)
end

for f in juliafiles
    cmd = `$julia_exe -p $(nprocs-1) --startup-file=no $(joinpath(testdir, f))`
    println("Executing $cmd")
    run(cmd)
end
