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

function plotcols(data, xcol, ycol; kwargs...)
  sorted = sortrows(data[:,[xcol,ycol]])
  x = sorted[:,1]
  y = sorted[:,2]
  plot!(log2.(x), y*1000; xticks=(log2.(x), string.(Int.(x))), kwargs...)
  return (x,y)
end

function plot_comparison(darray_filt, mpi_filt, serial_filt_time, filename, legpos)
  plot()
  plotcols(filter(ddim, 1, darray_filt), procs, mintime, label="DArray, rows", marker=:square)
  plotcols(filter(ddim, 2, darray_filt), procs, mintime, label="DArray, cols", marker=:square)
  plotcols(filter(ddim, 1, mpi_filt), procs, mintime, label="MPI, rows", marker=:circle)
  (ideal_x, y) = plotcols(filter(ddim, 2, mpi_filt), procs, mintime, label="MPI, cols", marker=:circle)
  hline!([serial_filt_time*1000], label="Base.Array")
  plot!(log2.(ideal_x), serial_filt_time*1000 ./ ideal_x, label="Ideal scaling", linestyle=:dash, xlabel="Number of processes", ylabel="Time (ms)", legend=legpos)
  savefig(filename)
end

plotlyjs()

plot_comparison(filter(threads,1,darray), filter(threads,1,mpi), minimum(filter(threads, 1, serial)[:,mintime]), "singlethread.svg", :top)
plot_comparison(filter(threads,32,darray), filter(threads,32,mpi), minimum(filter(threads, 32, serial)[:,mintime]), "multithread.svg", :topright)


