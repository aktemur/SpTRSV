#define ONELINE \
  __asm__("movslq (%rsi,%rax,4), %r9");  \
  __asm__("movsd (%rdx,%rax,8), %xmm1"); \
  __asm__("incq %rax");                  \
  __asm__("mulsd (%rcx,%r9,8), %xmm1");  \
  __asm__("subsd %xmm1, %xmm0");         \


#define BODY_5    ONELINE ONELINE ONELINE ONELINE ONELINE
#define BODY_25   BODY_5 BODY_5 BODY_5 BODY_5 BODY_5
#define BODY_125  BODY_25 BODY_25 BODY_25 BODY_25 BODY_25
#define BODY_500  BODY_125 BODY_125 BODY_125 BODY_125
#define BODY_2500 BODY_500 BODY_500 BODY_500 BODY_500 BODY_500
#define BODY_10K  BODY_2500 BODY_2500 BODY_2500 BODY_2500

void csrLenForwardSolve(int* __restrict rowPtr, int* __restrict colIndices, double* __restrict values,
                        double* __restrict x, double* __restrict b, unsigned int N) {
  // parameters %rdi, %rsi, %rdx, %rcx, %r8, %r9
  // The last parameter, N, is not needed and hence ignored.
  // Therefore we use %r9 for temporary values.
  __asm__("push %rax");
  __asm__("push %rbx");
  __asm__("push %rcx");
  __asm__("push %rdx");
  __asm__("push %rdi");
  __asm__("push %rsi");
  __asm__("push %r8");
  __asm__("push %r9");
  __asm__("push %r10");

  __asm__("xorl %eax, %eax"); // k <- 0
  __asm__("xorl %ebx, %ebx"); // i <- 0
  __asm__("jmp init");

  // Unrolled loop body begins
  __asm__("movslq (%rsi,%rax,4), %r9"); // r9 <- cols[k]
  __asm__("movsd (%rdx,%rax,8), %xmm1");// xmm1 <- vals[k]
  __asm__("incq %rax");                 // k++
  __asm__("mulsd (%rcx,%r9,8), %xmm1"); // xmm1 <- xmm1 * v[r9]
  __asm__("subsd %xmm1, %xmm0");        // sum <- sum - xmm1

  BODY_25
  // L_0:

  __asm__("divsd (%rdx,%rax,8), %xmm0");// xmm0 <- xmm0 / values[k] ; 5 bytes
  __asm__("movsd %xmm0, (%rcx,%rbx,8)");// x[i] <- xmm0             ; 5 bytes
  __asm__("incq %rax");                 // k++                      ; 3 bytes
  __asm__("incq %rbx");                 // i++                      ; 3 bytes

  // init:
  __asm__("init:");
  __asm__("movsd (%r8,%rbx,8), %xmm0"); // sum <- b[i]       ; 6 bytes
  __asm__("movslq (%rdi,%rbx,4), %r9"); // r9 <- rows[i]     ; 4 bytes
  __asm__("leaq -33(%rip), %r10");      // r10 <- &&L_0      ; 7 bytes  ;  L_0 is (5+5+3+3)+(6+4+7)=33 bytes behind
  __asm__("addq %r9, %r10");            // r10 <- r10 + r9
  __asm__("jmp *%r10");

  // end:
  __asm__("pop %r10");
  __asm__("pop %r9");
  __asm__("pop %r8");
  __asm__("pop %rsi");
  __asm__("pop %rdi");
  __asm__("pop %rdx");
  __asm__("pop %rcx");
  __asm__("pop %rbx");
  __asm__("pop %rax");
}

