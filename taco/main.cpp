#include "bench.h"

void sketch_alpha_blending();
void brighten_bench();
void mri_bench();
void movie_alpha_bench();
void movie_mask_bench();
void movie_subtitle_bench();
void movie_brighten_bench();
void bench_spmv();
void movie_decompress_bench();
void movie_compress_bench_brighten();
void movie_compress_bench_subtitle();
void movie_lz77_rle();
void movie_rle_lz77_bench_brighten();
void movie_rle_lz77_bench_subtitle();
void movie_mask_mul_bench();

int main(){
    auto bench = getEnvVar("BENCH");
    if (bench == "sketch"){
        sketch_alpha_blending();
    } else if (bench == "brighten"){
        brighten_bench();
    } else if (bench == "mri"){
        mri_bench();
    } else if (bench == "alpha"){
        movie_alpha_bench();
    } else if (bench == "mask"){
        movie_mask_bench();
    } else if (bench == "subtitle"){
        movie_subtitle_bench();
    } else if (bench == "mbrighten"){
        movie_brighten_bench();
    } else if (bench == "spmv"){
        bench_spmv();
    } else if (bench == "decompress"){
        movie_decompress_bench();
    } else if (bench == "brighten_compress"){
        movie_compress_bench_brighten();
    } else if (bench == "subtitle_compress"){
        movie_compress_bench_subtitle();
    } else if (bench == "lz77_rle"){
        movie_lz77_rle();
    } else if (bench == "lz77_rle_brighten"){
        movie_rle_lz77_bench_brighten();
    } else if (bench == "lz77_rle_subtitle"){
        movie_rle_lz77_bench_subtitle();
    } else if (bench == "maskmul"){
        movie_mask_mul_bench();
    }
    return 0;
}