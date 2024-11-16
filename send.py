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
    k = (int(l[2])-int(l[1]))//2
    v = (int(l[4])-int(l[3]))//2
    m = int(l[-1])
    ml = 10*['-1',]+[.9,.8,.6,.4,.2,.1]
    ml[:7]=['floor','f+cali','ceil','c+ceil','55/45','44/55','NONE',]
    
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            s=f'( {k},{v} ), {m}\n{ml[m]}\n\n{output_path}\n'
            file_content = file.read()
            data = json.loads(file_content)
            ol = []
            kl = list(data.keys())
            vl = list(data.values())
            avg=0.0
            tl=['tre','zh','news']
            for i in range(len(vl)):
                vl[i] = int(100*float(vl[i]))/100
                avg+=vl[i]
                for j in range(len(tl)):
                    if tl[j] in kl[i]:
                        ol.append(i)
            for i in ol:
                s=f'{s}\n"{kl[i]}": {vl[i]}'
            avg=int(100*avg/len(ol))/100
            s=f'{s}\n"average": {avg}'
        # send_qq_email(f'{output_path}: Success',f'{k},{v},{m}\n{ml[m]}\n\n{output_path}\n'+file_content)
        send_qq_email(f'{output_path}: Success',f'{s}\n\n\n{file_content}')
        print(f'{output_path}: Success')
    else:
        send_qq_email(f'{output_path}: Fail',f'{output_path}: Fail')
        print(f'{output_path}: Fail')
    