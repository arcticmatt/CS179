===== IO and Parsing Analysis =====
This analysis was done with a batch size of 2048.
Running the code regularly gave us a time of
16.841280 seconds.
Running the code with all the CUDA code commented out, effectively
leaving the computation time dominated by IO and parsing, gave us 
a time of 
15.630030 seconds.
The ratio of these numbers is 
0.92807850709.
This means ~93% of our time is spent doign IO and parsing, and only ~7% 
of our time is spent in the kernel.


===== Latency and Throughput (by batch size) =====
Note: measure latency by commenting out only the kernel.

1
with kernel: doesn’t finish
without kernel: Total time to run classify: 33.802284 (s)
latency: N/A
throughput: N/A

32
with kernel: Total time to run classify: 28.812044 (s)
without kernel: Total time to run classify: 17.505116 (s)
latency = 11.306928
throughput = 2.830123

1024
with kernel: Total time to run classify: 18.261539 (s)
without kernel: Total time to run classify: 16.133404 (s)
latency = 2.128135
throughput = 481.172482

2048
with kernel: Total time to run classify: 17.336798 (s)
without kernel: Total time to run classify: 15.755376 (s)
latency = 1.581422
throughput = 1295.036998

16384
with kernel: Total time to run classify: 17.078381 (s)
without kernel: Total time to run classify: 16.585133 (s)
latency = 0.493248
throughput = 33216.556377

65536
with kernel: Total time to run classify: 16.217661 (s)
without kernel: Total time to run classify: 15.914539 (s)
latency = 0.303122
throughput = 216203.376858
