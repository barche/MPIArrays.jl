open("out-elemental.txt", "w") do outf
  write(outf, "# nbprocs nbthreads distributed_dimension mintime (s) maxtime (s)\n")
  for fname in ARGS
    open(fname) do f
      nthreads = 0
      ntasks = 0
      timing = 0.0
      readnext = false
      for ln in eachline(f)
        if readnext
          timing = parse(Float64, split(ln)[2])
          break
        end
        if startswith(ln, "OPENBLAS_NUM_THREADS")
          nthreads = parse(Int,split(ln)[2])
        end
        if startswith(ln, "TASKS")
          ntasks = parse(Int,split(ln)[2])
        end
        if contains(ln, "Field=double")
          readnext = true
        end
      end
      write(outf, "$ntasks $nthreads 1 $timing $timing\n")
    end
  end
end