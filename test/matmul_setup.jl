using Random

const nb_threads = parse(Int,get(ENV, "OPENBLAS_NUM_THREADS", "-1"))
if nb_threads == -1
  error("Matmul tests require explicitly setting OPENBLAS_NUM_THREADS")
end

const nsamples = 4

function timed_mul(A,x)
  times = zeros(nsamples)
  for i in 1:length(times)
    val, times[i], bytes, gctime, memallocs = @timed b = A*x
  end
  return extrema(times[2:end])
end

function write_timings(filename, distributed_dimension, mintime, maxtime)
  filepath = filename
  if !isabspath(filename)
    filepath = joinpath(@__DIR__, filename)
  end
  open(filepath, "a") do f
    if position(f) == 0
      write(f, "# nbprocs nbthreads distributed_dimension mintime (s) maxtime (s)\n")
    end
    write(f, "$nb_procs $nb_threads $distributed_dimension $mintime $maxtime\n")
  end
end

const N = 5000

if do_serial
  const As = rand(N,N)
  const xs = rand(N)

  const ref = As*xs

  (mintime, maxtime) = timed_mul(As,xs)
  write_timings("out-serial.txt", 1, mintime, maxtime)
end
