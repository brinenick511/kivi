# {model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_ \
# residual{model_args.residual_length}
# 
# Mistral-7B-Instruct-v0.2_31500_2bits_group32_residual128
# from process_args import process_args

def work(model_name, model_path, k_bits, v_bits, group_size, residual_length, annotation):
    if model_name is None:
        if model_path is None:
            return 'ERROR'
        model_name = model_path.split("/")[-1]
    s = f'{model_name}_{k_bits}bits'
    # s += f'_g{group_size}_r{residual_length}'
    # s += f'_{annotation}'
    return s

if __name__ == '__main__':
    a=1
    # model_args, data_args, training_args = process_args()
    # model_name = model_args.model_name_or_path.split("/")[-1]
    # output_path = work(
    #     model_name,None,model_args.k_bits,model_args.v_bits,
    #     model_args.group_size,model_args.residual_length,None)
    # work()