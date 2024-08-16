from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def process_files(directory):
    results = []
    total = 0

    for filename in os.listdir(directory):
        if total == 40:
            break
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) >= 3:
                    total += 1
                    last_line = lines[-1].strip()
                    try:
                        if not args.other:
                            value = int(last_line.split()[-1])
                            value = value * 5
                        else:
                            value = int(last_line.split()[-1])
                    except (IndexError, ValueError):
                        continue

                    results.append(value)
    print(total)
    return results

parser = argparse.ArgumentParser(description='Calculate the AUC.')
parser.add_argument('--type', default='v2')
parser.add_argument('--attack', default='db_1')
parser.add_argument('--other_type', default='Random-v1-2')
parser.add_argument('--other', action='store_true')
parser.add_argument('--img_processing', action='store_true')
args = parser.parse_args()

if args.img_processing:
    if not args.other:
        directory_path = f'./log/v1.5/attack/jpeg/LA-{args.type}/{args.attack}'
    else:
        directory_path = f'./log/v1.5/attack/jpeg/{args.other_type}/{args.attack}'
else:
    if not args.other:
        directory_path = f'./log/v1.5/LA-{args.type}/{args.attack}'
    else:
        directory_path = f'./log/v1.5/{args.other_type}/{args.attack}'
results = process_files(directory_path)

average = np.mean(np.array(results))

print(average)