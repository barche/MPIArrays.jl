
function timed_mul(A,x)
  for i in 1:3
      @time b = A*x
  end
end

const N = 5000

if do_serial
  const As = rand(N,N)
  const xs = rand(N)

  const ref = As*xs

  println("Serial A*x:")
  timed_mul(As,xs)
end
