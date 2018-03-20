using Compat
import Compat.Sys: BINDIR

julia_exe = joinpath(BINDIR, Base.julia_exename())
mpiexec = "mpiexec"

testdir = dirname(@__FILE__)

mpifiles = ["basics.jl", "matmul_mpi.jl"]
juliafiles = ["matmul_darray.jl"]
nprocs = clamp(Sys.CPU_CORES ÷ 2, 2, 4)
ENV["OPENBLAS_NUM_THREADS"] = 1

for f in mpifiles
    cmd = `$mpiexec -n $nprocs $julia_exe $(joinpath(testdir, f))`
    println("Executing $cmd")
    run(cmd)
end

for f in juliafiles
    cmd = `$julia_exe -p $(nprocs-1) $(joinpath(testdir, f))`
    println("Executing $cmd")
    run(cmd)
end
