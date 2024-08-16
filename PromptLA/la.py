import numpy as np
from statsmodels.stats.weightstats import ttest_ind, ztest
import random
from Bayesian_KL import *

def statistical_inference(sequence_1=[], sequence_2=[], delta=0.01):
    # Check if the lengths of the sequences are not equal
    if len(sequence_1) != len(sequence_2):
        return None
    
    # If the length of sequences is less than 30, use t-test from statsmodels
    if len(sequence_1) < 30:
        t_stat, p_value, _ = ttest_ind(sequence_1, sequence_2, usevar='unequal')
    # If the length of sequences is 30 or greater, use z-test from statsmodels
    else:
        t_stat, p_value = ztest(sequence_1, sequence_2)
    
    # Determine if the difference is significant based on the delta threshold
    logging.info(p_value)
    return p_value < delta


def get_dataloader_new(folder_path, sub_list, batch_size=20, num_workers=0, attack=False):
    if attack:
        dataset = ImageFolderDataset(folder_path, transform_attack)
    else:
        dataset = ImageFolderDataset(folder_path, transform)

    # sub_test = random.sample(range(len(dataset)), int(fp_num))
    sub_test = sub_list
    # print(sub_test)
    dataset = torch.utils.data.Subset(dataset, sub_test)
    # print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader


def KL_get_feedback(action, sub_list1, sub_list2):

    folder_path1 = f'./data/trigger/v1.5/prompt{action}/test_true/v1.5'
    # folder_path1 = f'./data/trigger/prompt{action}/test_false/db_4'
    # dataloader1 = get_dataloader(folder_path1, num_iter * la_args.img_num, attack=la_args.img_processing)
    dataloader1 = get_dataloader_new(folder_path1, sub_list1, attack=la_args.img_processing)
    features1 = get_features(dataloader1, model, device)
    
    folder_path2 = f'./data/trigger/v1.5/prompt{action}/train_true/v1.5'
    # folder_path2 = f'./data/trigger/prompt{action}/test_false/db_4'
    # dataloader2 = get_dataloader(folder_path2, num_iter * la_args.img_num, attack=la_args.img_processing)
    dataloader2 = get_dataloader_new(folder_path2, sub_list1, attack=la_args.img_processing)
    features2 = get_features(dataloader2, model, device)

    if la_args.test_FP:
        folder_path3 = f'./data/trigger/v1.5/prompt{action}/train_true/v1.5'
    else:
        folder_path3 = f'./data/trigger/v1.5/prompt{action}/test_false/{la_args.attack}'
        # folder_path3 = f'./data/trigger/v1.4/prompt{action}/test_true/{la_args.attack}'
    # dataloader3 = get_dataloader(folder_path3, num_iter * la_args.img_num, attack=la_args.img_processing)
    dataloader3 = get_dataloader_new(folder_path3, sub_list2, attack=la_args.img_processing)
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

def sequential_la(action_set):
    cnt = 0
    num_actions = len(action_set)
    # initialize the feedback sequence
    feedback_sequence = {action: [] for action in action_set}
    # initialize the sub_list and left list
    sub_list1 = {action: [] for action in action_set}
    left_list1 = {action: list(range(100)) for action in action_set}
    sub_list2 = {action: [] for action in action_set}
    left_list2 = {action: list(range(100)) for action in action_set}
    # initialize the average reward
    average_reward = {action: 0 for action in action_set}
    # initialize the candidate flag
    candidate_flag ={action: True for action in action_set}
    num_iter = 0
    while sum(candidate_flag.values()) > 1:
        num_iter += 1

        logging.info(f'Num_iter: {num_iter}')

        for action in action_set:
            if candidate_flag[action]:
                cnt += 1

                for i in range(la_args.img_num):
                    idx = random.sample(left_list1[action], 1)[0]
                    sub_list1[action].append(idx)
                    left_list1[action].remove(idx)

                    idx = random.sample(left_list2[action], 1)[0]
                    sub_list2[action].append(idx)
                    left_list2[action].remove(idx)
                
                # get the feedback
                # feedback = KL_get_feedback(action, num_iter)
                '''
                if num_iter == 1:
                    feedback = KL_get_feedback(action, sub_list1[action], sub_list2[action])
                    feedback_sequence[action].append(feedback)
                else:
                    for i in range(min(5, num_iter - 1)):
                        sub_list1_random = random.sample(sub_list1[action][:-la_args.img_num], (num_iter - 2) * la_args.img_num) + sub_list1[action][-la_args.img_num:]
                        sub_list2_random = random.sample(sub_list2[action][:-la_args.img_num], (num_iter - 2) * la_args.img_num) + sub_list2[action][-la_args.img_num:]
                        feedback = KL_get_feedback(action, sub_list1_random, sub_list2_random)
                        feedback_sequence[action].append(feedback)
                '''

                for i in range(min(5, num_iter)):
                    sub_list1_random = random.sample(sub_list1[action][:-la_args.img_num], (num_iter - i - 1) * la_args.img_num) + sub_list1[action][-la_args.img_num:]
                    sub_list2_random = random.sample(sub_list2[action][:-la_args.img_num], (num_iter - i - 1) * la_args.img_num) + sub_list2[action][-la_args.img_num:]
                    feedback = KL_get_feedback(action, sub_list1_random, sub_list2_random)
                    feedback_sequence[action].append(feedback)
                
                # feedback = KL_get_feedback(action, sub_list1[action], sub_list2[action])
                # feedback_sequence[action].append(feedback)

                # update the average reward
                average_reward[action] = np.mean(feedback_sequence[action])
             
        logging.info(feedback_sequence)
        logging.info(average_reward)
        logging.info(candidate_flag)

        if num_iter >= la_args.start_iter: # avoid cold start
            # get the estimated best action
            best_action = max(average_reward, key=average_reward.get)

            # trick1
            if average_reward[best_action] < 0.05:
                break

            # update the candidate flag

            min_average_reward = 1e5

            for action in action_set:
                if candidate_flag[action] and action != best_action:
                    min_average_reward = min(min_average_reward, average_reward[action])
                    if statistical_inference(feedback_sequence[best_action], feedback_sequence[action], delta=la_args.alpha):
                        candidate_flag[action] = False
                        average_reward[action] = 0
            
             # trick2
            if min_average_reward >= la_args.threshold:
                break

        if num_iter >= la_args.end_iter: # avoid infinite loop
            break
    # return the estimated best action
    logging.info(cnt)
    best_action = max(average_reward, key=average_reward.get)
    return cnt, best_action, average_reward[best_action]

'''
action_set = []
file_name = './cmd/seed_words_list.txt'
with open(file_name, 'r', encoding='utf-8') as file:
    for line in file:
        action_set.append(line.strip())
'''

la_parser = argparse.ArgumentParser(description='la.')
la_parser.add_argument('--cuda', default='cuda:0')
la_parser.add_argument('--attack', default='db_1')
la_parser.add_argument('--img_num', default=10, type=int)
la_parser.add_argument('--start_iter', default=5, type=int)
la_parser.add_argument('--end_iter', default=10, type=int)
la_parser.add_argument('--test_FP', action='store_true')
la_parser.add_argument('--img_processing', action='store_true')
la_parser.add_argument('--start_id', default=1, type=int)
la_parser.add_argument('--end_id', default=5, type=int)
la_parser.add_argument('--threshold', default=0.25, type=float)
la_parser.add_argument('--alpha', default=0.01, type=float)
la_parser.add_argument('--runname', default='LA-v2')
la_args = la_parser.parse_args()

cwd = os.getcwd()
if la_args.img_processing:
    if la_args.test_FP:
        log_dir = os.path.join(cwd, 'log', 'v1.5', 'attack', 'crop', la_args.runname, 'baseline')
    else:
        log_dir = os.path.join(cwd, 'log', 'v1.5', 'attack', 'crop', la_args.runname, la_args.attack)
else:
    if la_args.test_FP:
        log_dir = os.path.join(cwd, 'log', 'v1.5', la_args.runname, 'baseline')
    else:
        log_dir = os.path.join(cwd, 'log', 'v1.5', la_args.runname, la_args.attack)
os.makedirs(log_dir, exist_ok=True)
logfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + f'result_{la_args.img_num}_{la_args.start_iter}_{la_args.end_iter}.txt')
set_up_logger(logfile)
'''
save_dir = os.path.join(cwd, 'result', 'v1.5', 'LA')
os.makedirs(save_dir, exist_ok=True)
if la_args.test_FP:
    file_path = os.path.join(save_dir, 'baseline.txt')
else:
    file_path = os.path.join(save_dir, f'{la_args.attack}.txt')
'''
device = torch.device(la_args.cuda if torch.cuda.is_available() else "cpu")
model = get_inception_model().to(device)

def main():
    
    numbers = list(range(1, 51))

    flag = False
    num_img = 0
    best_reward = 0

    while len(numbers) > 0:
        action_set = random.sample(numbers, 5)

        for number in action_set:
            numbers.remove(number)
        
        logging.info(action_set)
        logging.info(numbers)

        cnt, best_action, average_reward = sequential_la(action_set)
        num_img += cnt
        logging.info((best_action, average_reward))

        if average_reward >= la_args.threshold:
            flag = True
            logging.info("successfully find")
            logging.info(num_img)
            break
        else:
            best_reward = max(best_reward, average_reward)
    if not flag:
        logging.info("failure")
        logging.info(best_reward)
        logging.info(num_img)

if __name__ == "__main__":
    main()