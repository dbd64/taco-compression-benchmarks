#include "bench.h"

void sketch_alpha_blending();
void brighten_bench();
void mri_bench();
void movie_alpha_bench();
void movie_mask_bench();
void movie_subtitle_bench();
void movie_brighten_bench();

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
    }
    return 0;
}