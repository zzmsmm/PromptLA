from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import os
import argparse

def process_files(directory):
    results = []
    success = 0
    total = 0

    for filename in os.listdir(directory):
        if total == 20:
            break
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) >= 3:
                    total += 1
                    second_last_line = lines[-2].strip()
                    third_last_line = lines[-3].strip()

                    if second_last_line.endswith("successfully find"):
                        success += 1
                        try:
                            if not args.other:
                                value = float(third_last_line.split('(')[-1].split(',')[1].strip(')'))
                            else:
                                value = float(third_last_line.split()[-1])
                        except (IndexError, ValueError):
                            continue
                    else:
                        try:
                            value = float(second_last_line.split()[-1])
                        except (IndexError, ValueError):
                            continue

                    results.append(value)
    print(total)
    return results, 1.0 * success / total

parser = argparse.ArgumentParser(description='Calculate the AUC.')
parser.add_argument('--type', default='v1')
parser.add_argument('--attack', default='db_1')
parser.add_argument('--other_type', default='Random-v2-2')
parser.add_argument('--other', action='store_true')
parser.add_argument('--img_processing', action='store_true')
parser.add_argument('--baseline', action='store_true')
args = parser.parse_args()

if args.baseline:
    with open(f'./log/v1.5/baseline/f/all/all-{args.attack}.txt', 'r') as file:
        lines = file.readlines()
    results_0 = []
    results_1 = []
    for line in lines:
        num1, num2 = map(float, line.split())
        results_0.append(num1)
        results_1.append(num2)
    results = results_1 + results_0
    zeros_array = [0] * len(results_0)
    ones_array = [1] * len(results_1)
    y_true = zeros_array + ones_array
    y_scores = results
    auc = roc_auc_score(y_true, y_scores)
    print(f"AUC: {auc}")
else:
    if args.img_processing:
        if not args.other:
            directory_path_0 = f'./log/v1.5/attack/crop/LA-{args.type}/baseline'
        else:
            directory_path_0 = f'./log/v1.5/attack/crop/{args.other_type}/baseline'
    else:
        if not args.other:
            directory_path_0 = f'./log/v1.5/LA-{args.type}/baseline'
        else:
            directory_path_0 = f'./log/v1.5/{args.other_type}/baseline'

    results_0, acc = process_files(directory_path_0)
    print("Baseline ACC: %.3f" % acc)

    if args.img_processing:
        if not args.other:
            directory_path_1 = f'./log/v1.5/attack/crop/LA-{args.type}/{args.attack}'
        else:
            directory_path_1 = f'./log/v1.5/attack/crop/{args.other_type}/{args.attack}'
    else:
        if not args.other:
            directory_path_1 = f'./log/v1.5/LA-{args.type}/{args.attack}'
        else:
            directory_path_1 = f'./log/v1.5/{args.other_type}/{args.attack}'
    results_1, acc = process_files(directory_path_1)
    print("%s ACC: %.3f" % (args.attack, acc))

    zeros_array = [0] * len(results_0)
    ones_array = [1] * len(results_1)
    y_true = zeros_array + ones_array
    # y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    y_scores = results_0 + results_1



    auc = roc_auc_score(y_true, y_scores)
    print(f"AUC: {auc}")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    if args.img_processing:
        if not args.other:
            save_dir = f'./result/v1.5/attack/crop/AUC-{args.type}'
        else:
            save_dir = f'./result/v1.5/attack/crop/AUC-{args.other_type}'
    else:
        if not args.other:
            save_dir = f'./result/v1.5/AUC-{args.type}'
        else:
            save_dir = f'./result/v1.5/AUC-{args.other_type}'
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f'{save_dir}/AUC-{args.attack}.png', dpi=300, bbox_inches='tight')
