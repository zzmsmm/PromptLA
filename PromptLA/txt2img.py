from diffusers import DiffusionPipeline
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid
import os
from compel import Compel

def read_specific_line(file_name, line_number):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            for current_line_number, line in enumerate(file, start=1):
                if current_line_number == line_number:
                    return line.strip()
        raise ValueError(f"Line number {line_number} exceeds the total number of lines in the file.")
    except FileNotFoundError:
        return "The file does not exist."


import argparse

parser = argparse.ArgumentParser(description='Generate samples through models.')
parser.add_argument('--train_num', default=0, type=int)
parser.add_argument('--test_num', default=0, type=int)
parser.add_argument('--prompt', default='')
# parser.add_argument('--prompt_id', default='1')
parser.add_argument('--prompt_id', default=1, type=int, help='prompt id')
parser.add_argument('--dreambooth', action='store_true')
parser.add_argument('--dreambooth_type', default='')
parser.add_argument('--dreamlike', action='store_true')
parser.add_argument('--cuda', default='cuda:2')

args = parser.parse_args()

if args.dreamlike:
    pipeline = DiffusionPipeline.from_pretrained("./model/dreamlike1.0", torch_dtype=torch.float16)
else:
    pipeline = DiffusionPipeline.from_pretrained("./model/pretrained1.5", torch_dtype=torch.float16)

if args.dreambooth:
    pipeline.load_lora_weights(f"./dreambooth/v1.5/{args.dreambooth_type}")

pipeline.to(args.cuda)

# generator = torch.Generator("cuda:2").manual_seed(50)
file_name = './cmd/seed_words_list.txt'
prompt = read_specific_line(file_name, args.prompt_id)
print(prompt)
# prompt = args.prompt
# negative_prompt = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w"

if args.dreambooth:
    output_dir_train = f'./data/trigger/v1.5/prompt{args.prompt_id}/train_false/db_{args.dreambooth_type}'
    output_dir_test = f'./data/trigger/v1.5/prompt{args.prompt_id}/test_false/db_{args.dreambooth_type}'
elif args.dreamlike:
    output_dir_train = f'./data/trigger/v1.5/prompt{args.prompt_id}/train_false/dl'
    output_dir_test = f'./data/trigger/v1.5/prompt{args.prompt_id}/test_false/dl'
else:
    output_dir_train = f'./data/trigger/v1.5/prompt{args.prompt_id}/train_true/v1.5'
    output_dir_test = f'./data/trigger/v1.5/prompt{args.prompt_id}/test_true/v1.5'

os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)

for i in range(args.train_num):
    out_images = pipeline(prompt).images
    out_images[0].save(os.path.join(output_dir_train, "img" + str(i) + ".png"))
for i in range(args.test_num):
    out_images = pipeline(prompt).images
    out_images[0].save(os.path.join(output_dir_test, "img" + str(i) + ".png"))

'''
out_images = pipeline2(prompt, negative_prompt=negative_prompt, generator=generator).images
out_images[0].save("output1.png")
'''