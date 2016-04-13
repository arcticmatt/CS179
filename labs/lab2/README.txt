Question 1.1: Latency Hiding
----------------------------
It takes approximately 10 arithmetic instructions to hide the latency of a single
arithmetic instruction on a GK110. This is because the latency for arithmetic
instructions on a GK110 is ~10 ns, and the GPU clock is 1 GHz (1 clock/ns). If
we have 10 arithmetic instructions, the GPU will be able to execute them all
in parallel after a ~10 second delay, giving us a rate of ~1 instruction/clock
cycle.

Question 1.2: Thread Divergence
-------------------------------
(a)
This code diverges because of the conditional statement. The conditional causes
a difference in behavior in threads based on the calculated idx. Thus, since
threads must run the same set of instructions at the same time, threads must
execute the different instructions in serial. In other words, we get warp
divergence.

(b)
This code does not diverge. Although each thread will execute a different number
of instructions (e.g. threadIdx.x will be different), they will all execute
in parallel, since the instruction in the for-loop is the same. There is
some "divergent-like" behavior, because some threads will finish earlier than
others. For example, the thread where threadIdx.x = 1 will finish earlier than
the thread with threadIdx.x = 31. However, its instructions (while running)
will still be in parallel with the other threads.

Question 1.3: Coalesced Memory Access
-------------------------------------
(a)
This write is coalesced, and writes to 1 128 byte cache line.

(b)
This write is not coalesced, and writes to 32 128 byte cache lines.

(c)
This write is not coalesced, and writes to 2 128 byte cache lines.

Question 1.4: Bank Conflicts and Instruction Dependencies
---------------------------------------------------------
(a)
There are no bank conflicts. This is because each thread (within each warp)
accesses a separate row of lhs; since lhs is stored in column major format,
each row of lhs is stored in a separate bank. Then, each thread (within
each warp) has a constant j, which means each thread will access the same
rows and column of rhs (causing the bank variables to be broadcast). Thus,
there are no bank conflicts.

(b)
a0 = lhs[i + 32 * k]
b0 = rhs[k + 128 * j]
c0 = output[i + 32 * j]
d0 = FMA(a0, b0, c0)
output[i + 32 * j] = d0

a1 = lhs[i + 32 * (k + 1)]
b1 = rhs[(k + 1) + 128 * j]
c1 = output[i + 32 * j]
d1 = FMA(a1, b1, c1)
output[i + 32 * j] = d1

(c)
d0 is dependent on a0, b0, and c0
the write to output is dependent on d0

the latter 5 instructions are dependent on the completion of the former 5
    instructions
d1 is dependent on a1, b1, and c1
the write to output is dependent on d1

(d)
int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    float output = output[i + 32 * j];
    float l1 = lhs[i + 32 * k];
    float l2 = lhs[i + 32 * (k + 1)];
    float r1 = rhs[k + 128 * j];
    float r2 = rhs[(k + 1) + 128 * j];
    output += l1 * r1;
    output += l2 * r2;
    output[i + 32 * j] = output;
}

// "pseudo-assembly" interpretation
output = output[i + 32 * j]
a0 = lhs[i + 32 * k]
b0 = rhs[k + 128 * j]
a1 = lhs[i + 32 * (k + 1)]
b1 = rhs[(k + 1) + 128 * j]
output = FMA(a0, b0, output)
output = FMA(a1, b1, output)
output[i + 32 * j] = output

(e)
We could unroll the for-loop even more. That is, we could handle more values
of k within each iteration (e.g. 4, 8, etc.), or just completely unroll
it and get rid of the for-loop (although this would look pretty bad and
take up a lot of space).

BONUS (+5 points, maximum set score is 100 even with bonus)
-----------------------------------------------------------
- It requires two operations of order N as opposed to one (two for-loops).
- It requires an unnecessary loads since it writes an intermediate result to
    a before computing the final answer. That is, there are four loads in
    total (2 for each for-loop) as opposed to the 3 as seen in the faster
    method.
- Additional overhead of method calling, passing parameters (CDECL).
