import pynvml
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import sys
import argparse
from tqdm import tqdm
import os
import json
import csv
from filelock import FileLock

import dei_utils as dei

from utils.process_args import process_args, define_path

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    # parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    # parser.add_argument('--t', action='store_true', help="Test for short")
    # parser.add_argument('--l', action='store_true', help="Test for long")
    # parser.add_argument('--m', action='store_true', help="Multi-GPUs")
    return parser.parse_args(args)

def send_qq_email(subject='### GPU提醒 ###', body='<=GPU提醒=>'):
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = '2057807259@qq.com'
    msg['To'] = '2747350438@qq.com'

    smtp_server = 'smtp.qq.com'
    smtp_port = 587
    sender_email = '2057807259@qq.com'
    password = 'pobnlymirwhmciba'

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, [msg['To']], msg.as_string())
        print('邮件发送成功')
        # sys.exit(0)
    except smtplib.SMTPException as e:
        print('邮件发送失败:', str(e))
        
if __name__ == "__main__":
    # args = parse_args()
    # model_name = args.model
    # path = f'/new_data/yanghq/LongBench/pred/{model_name}/result.json'
    model_args, data_args, training_args = process_args()
    model_name = model_args.model_name_or_path.split("/")[-1]
    output_path = define_path(
        model_name,None,model_args.k_bits,model_args.v_bits,
        model_args.group_size,model_args.residual_length,model_args.annotation)
    path = f'pred/{output_path}/result.json'
    
    l = model_args.annotation.split('_')
    
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            # s=f'q = ( {l[0]}, {l[1]} )\nm = ( {l[2]}, {l[3]} )\n\n{output_path}\n'
            # s=f'gamma = {l[-1]}\n\nasym-kv\n{output_path}\n'
            s=f'\n{l[-2]}, {l[-1]}\n\n{output_path}\n'
            file_content = file.read()
            # data = json.loads(file_content)
            # ol = []
            # kl = list(data.keys())
            # vl = list(data.values())
            # avg=0.0
            # tl=['tre','zh','news']
            # for i in range(len(vl)):
            #     vl[i] = int(100*float(vl[i]))/100
            #     avg+=vl[i]
            #     for j in range(len(tl)):
            #         if tl[j] in kl[i]:
            #             ol.append(i)
            # new_data = l
            # for i in ol:
            #     s=f'{s}\n"{kl[i]}": {vl[i]}'
            #     new_data+=[vl[i],]
            # avg=int(100*avg/len(ol))/100
            # s=f'{s}\n"average": {avg}'
            # if 'm' in model_args.annotation:
            #     file_name = '/new_data/yanghq/ans_m.csv'
            # elif 'q' in model_args.annotation:
            #     file_name = '/new_data/yanghq/ans_q.csv'
            # else:
            #     file_name = '/new_data/yanghq/ans.csv'
            # lock_file = file_name + '.lock'  # 锁文件名
            # lock = FileLock(lock_file)
            # with lock:
            #     try:
            #         with open(file_name, 'x', newline='') as ffile:
            #             writer = csv.writer(ffile)
            #             writer.writerow(['kq', 'vq', 'km', 'vm', 'trec', 'mqazh'])
            #     except FileExistsError:
            #         pass
            #     with open(file_name, 'a', newline='') as ffile:
            #         writer = csv.writer(ffile)
            #         writer.writerow(new_data)
        # send_qq_email(f'{output_path}: Success',f'{k},{v},{m}\n{ml[m]}\n\n{output_path}\n'+file_content)
        send_qq_email(f'{output_path}: Success',f'{s}\n\n\n{file_content}')
        print(f'{output_path}: Success')
    else:
        directory = f'pred/{output_path}/'
        s=f'{output_path}: Fail'
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            s+=(f"\nfile: {file}")
                            s+=(f"\nlines: {len(lines)}\n")
                    except Exception as e:
                        s+=(f"\nfail to read {file} : {e}")
        send_qq_email(f'{output_path}: Fail',s)
        print(f'{output_path}: Fail')
    