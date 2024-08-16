# random & average methods as baseline
import numpy as np
import random
from Bayesian_KL import *

def get_relative_KL(prompt_id):

    folder_path1 = f'./data/trigger/v1.5/prompt{prompt_id}/test_true/v1.5'
    # folder_path1 = f'./data/trigger/prompt{prompt_id}/test_false/db_4'
    # dataloader1 = get_dataloader(folder_path1, num_iter * ba_args.img_num, attack=ba_args.img_processing)
    dataloader1 = get_dataloader(folder_path1, ba_args.img_num)
    features1 = get_features(dataloader1, model, device)
    
    folder_path2 = f'./data/trigger/v1.5/prompt{prompt_id}/train_true/v1.5'
    # folder_path2 = f'./data/trigger/prompt{prompt_id}/test_false/db_4'
    # dataloader2 = get_dataloader(folder_path2, num_iter * ba_args.img_num, attack=ba_args.img_processing)
    dataloader2 = get_dataloader(folder_path2, ba_args.img_num)
    features2 = get_features(dataloader2, model, device)

    if ba_args.test_FP:
        folder_path3 = f'./data/trigger/v1.5/prompt{prompt_id}/train_true/v1.5'
    else:
        # folder_path3 = f'./data/trigger/v1.5/prompt{prompt_id}/test_false/{ba_args.attack}'
        folder_path3 = f'./data/trigger/v1.4/prompt{prompt_id}/test_true/{ba_args.attack}'
    # dataloader3 = get_dataloader(folder_path3, num_iter * ba_args.img_num, attack=ba_args.img_processing)
    dataloader3 = get_dataloader(folder_path3, ba_args.img_num)
    features3 = get_features(dataloader3, model, device)

    prior_mean = torch.zeros(features1.size(1)).to(device)
    prior_cov = torch.eye(features1.size(1)).to(device)
    prior_dof = features1.size(1)
    prior_scale = 1.0

    mean1, cov1 = bayesian_estimate(features1, prior_mean, prior_cov, prior_dof, prior_scale)
    mean2, cov2 = bayesian_estimate(features2, prior_mean, prior_cov, prior_dof, prior_scale)
    mean3, cov3 = bayesian_estimate(features3, prior_mean, prior_cov, prior_dof, prior_scale)

    kl_0 = kl_divergence(mean1, cov1, mean2, cov2, device)  
    kl_1 = kl_divergence(mean1, cov1, mean3, cov3, device) 

    return (kl_1 - kl_0) / kl_0

ba_parser = argparse.ArgumentParser(description='baseline.')
ba_parser.add_argument('--cuda', default='cuda:0')
ba_parser.add_argument('--attack', default='db_1')
ba_parser.add_argument('--img_num', default=50, type=int)
ba_parser.add_argument('--test_FP', action='store_true')
ba_parser.add_argument('--img_processing', action='store_true')
ba_parser.add_argument('--threshold', default=0.3, type=float)
ba_parser.add_argument('--runname', default='Average')
ba_parser.add_argument('--average', action='store_true')
ba_args = ba_parser.parse_args()

cwd = os.getcwd()
if ba_args.test_FP:
    log_dir = os.path.join(cwd, 'log', 'v1.5', ba_args.runname, 'baseline')
else:
    log_dir = os.path.join(cwd, 'log', 'v1.5', ba_args.runname, ba_args.attack)
os.makedirs(log_dir, exist_ok=True)
logfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + f'result.txt')
set_up_logger(logfile)

device = torch.device(ba_args.cuda if torch.cuda.is_available() else "cpu")
model = get_inception_model().to(device)

def main():
    numbers = list(range(1, 51))

    flag = False
    num_img = 0
    best_result = 0
    
    if not ba_args.average:
        while len(numbers) > 0:
            num_img += ba_args.img_num

            random_id = random.sample(numbers, 1)[0]

            logging.info(random_id)
            logging.info(numbers)
            
            numbers.remove(random_id)
            result = get_relative_KL(random_id)
            logging.info(result)
            best_result = max(best_result, result)

            if best_result >= ba_args.threshold:
                flag = True
                logging.info("successfully find")
                logging.info(num_img)
                break
        
        if not flag:
            logging.info("failure")
            logging.info(best_result)
            logging.info(num_img)
    else:
        total_result = 0
        while len(numbers) > 0:
            num_img += ba_args.img_num

            random_id = random.sample(numbers, 1)[0]

            logging.info(random_id)
            logging.info(numbers)
            
            numbers.remove(random_id)
            result = get_relative_KL(random_id)
            # logging.info(result)
            average_result = (average_result * (49 - len(numbers)) + result) / (50 - len(numbers))
            logging.info(result)
        
if __name__ == "__main__":
    main()