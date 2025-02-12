gpuid=0

anno_list=(32_32_32_32_0_0 16_32_32_32_0_0 32_16_32_32_0_0 32_32_16_32_0_0 32_32_32_16_0_0 16_16_32_32_0_0 32_16_16_32_0_0 32_32_16_16_0_0 16_32_32_16_0_0 0_32_32_32_0_0 32_0_32_32_0_0 32_32_0_32_0_0 32_32_32_0_0_0 )
anno_list=(32_32_32_32_kivi_0_0 32_0_32_32_asym_0_0 24_0_32_8_ours_1_1 24_0_32_8_ours_0_0 )
anno_list=(24_0_32_8_ours_1_1 24_0_32_8_ours_0_0 )
anno_list=(32_32_32_32_kivi_0_0 32_0_32_32_asym_0_0 )
anno_list=(32_32_32_32_kivi_0_0 32_0_32_32_asym_0_0 32_0_32_16_ours_0_0 32_0_32_16_ours_1_1 24_0_32_8_ours_1_1 24_0_32_8_ours_0_0 )

echo "numbers of array = ${#anno_list[*]}"

for anno in ${anno_list[@]}
do
    echo $anno
    NUMEXPR_MAX_THREADS=127 CUDA_VISIBLE_DEVICES=$gpuid python mem_spd_test.py ${anno}
done
