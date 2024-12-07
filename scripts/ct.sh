gpuid=9

anno_list=(32_32_32_32_0_0 16_32_32_32_0_0 32_16_32_32_0_0 32_32_16_32_0_0 32_32_32_16_0_0 16_16_32_32_0_0 32_16_16_32_0_0 32_32_16_16_0_0 16_32_32_16_0_0 0_32_32_32_0_0 32_0_32_32_0_0 32_32_0_32_0_0 32_32_32_0_0_0 )
# anno_list=(32_32_32_32_0_0 )

echo "numbers of array = ${#anno_list[*]}"

for anno in ${anno_list[@]}
do
    echo $anno
    CUDA_VISIBLE_DEVICES=$gpuid python mem_spd_test.py ${anno}
done
