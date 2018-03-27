using Plots

const serial = readdlm("out-serial.txt")
const darray = readdlm("out-darray.txt")
const mpi = readdlm("out-mpi.txt")

# column indices
const procs = 1
const threads = 2
const ddim = 3
const mintime = 4
const maxtime = 5

function filter(col,val,data)
  matching_rows = find(x -> x == val, data[:,col])
  result = similar(data, length(matching_rows), size(data,2))
  for (i,j) in enumerate(matching_rows)
    result[i,:] .= data[j,:]
  end
  return result
end

function plotcols(data, xcol, ycol, name)
  plot!(data[:,xcol], data[:,ycol], label=name)
end

darray_st = filter(threads,1,darray)
mpi_st = filter(threads,1,mpi)

plotcols(filter(ddim, 1, darray_st), procs, mintime, "DArray, rows")
plotcols(filter(ddim, 2, darray_st), procs, mintime, "DArray, cols")
plotcols(filter(ddim, 1, mpi_st), procs, mintime, "DArray, rows")
plotcols(filter(ddim, 2, mpi_st), procs, mintime, "DArray, cols")
savefig("test.pdf")