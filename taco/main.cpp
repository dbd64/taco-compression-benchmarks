//int main() {
//    return 0;
//}
//int foo () {
//  return 0;
//}

// #include "benchmark/include/benchmark/benchmark.h"
// BENCHMARK_MAIN();

void sketch_alpha_blending();
void brighten_bench();
void mri_bench();
void movie_alpha_bench();

int main(){
    // sketch_alpha_blending();
    // brighten_bench();
    // mri_bench();
    movie_alpha_bench();
    return 0;
}