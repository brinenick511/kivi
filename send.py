import pynvml
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import sys
import argparse
from tqdm import tqdm
import os

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
        sys.exit(0)
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
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        send_qq_email(f'{output_path}: Success',f'{output_path}\n'+file_content)
        print(f'{output_path}: Success')
    else:
        send_qq_email(f'{output_path}: Fail',f'{output_path}: Fail')
        print(f'{output_path}: Fail')
    