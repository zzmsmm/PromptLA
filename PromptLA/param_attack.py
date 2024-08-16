from diffusers import DiffusionPipeline
import torch
from diffusers import StableDiffusionPipeline
import os
import argparse

def read_specific_line(file_name, line_number):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            for current_line_number, line in enumerate(file, start=1):
                if current_line_number == line_number:
                    return line.strip()
        raise ValueError(f"Line number {line_number} exceeds the total number of lines in the file.")
    except FileNotFoundError:
        return "The file does not exist."

parser = argparse.ArgumentParser(description='Generate false samples through parameter attack.')
parser.add_argument('--a', default=0.001, type=float, help='random rate')
parser.add_argument('--key_name', default='attentions', help='key module name')
parser.add_argument('--attack_type', default='pa_1', help='key module name')
parser.add_argument('--start_id', default=0, type=int, help='start index')
parser.add_argument('--end_id', default=100, type=int, help='end index')
parser.add_argument('--prompt_id', default=50, type=int, help='prompt id')
parser.add_argument('--prompt', default='attentions')
parser.add_argument('--cuda', default='cuda:0')

args = parser.parse_args()

pipeline1 = DiffusionPipeline.from_pretrained("./model/pretrained1.5", torch_dtype=torch.float16)
pipeline2 = DiffusionPipeline.from_pretrained("./model/pretrained1.5", torch_dtype=torch.float16)

pipeline1.to(args.cuda)
pipeline2.to(args.cuda)

# text_encoder, unet, vae
state_dict1 = pipeline1.unet.state_dict()
state_dict2 = pipeline2.unet.state_dict()

# key_name = 'attn'
# a = 0.001

for param_name, param2 in state_dict2.items():
    print(param_name)
    if args.key_name in param_name:
        param2 += args.a * torch.rand_like(param2)

'''
euclidean_distance = 0
cosine_similarity = 0
cnt = 0

for ((param_name1, param1), (param_name2, param2)) in zip(state_dict1.items(), state_dict2.items()):
    if param_name1 == param_name2:
        cnt += 1
        euclidean_distance += torch.norm(param1 - param2).item()
        
        param1_float = torch.flatten(param1.float())
        param2_float = torch.flatten(param2.float())
        cosine_similarity += (torch.dot(param1_float, param2_float) / (torch.norm(param1_float) * torch.norm(param2_float))).item()

print(cnt)
print(euclidean_distance)
print(cosine_similarity / cnt)


output_dir = f'./data/trigger/v1.5/prompt{args.prompt_id}/'

os.makedirs(os.path.join(output_dir, 'test_false', args.attack_type), exist_ok=True)

# Example usage
file_name = './cmd/seed_words_list.txt'
prompt = read_specific_line(file_name, args.prompt_id)
# print(prompt)


for i in range(args.start_id, args.end_id):
    out_images = pipeline2(prompt).images
    out_images[0].save(os.path.join(output_dir, 'test_false', args.attack_type, "img" + str(i) + ".png"))
'''

'''
generator = torch.Generator("cuda:2").manual_seed(50)
out_images = pipeline2(prompt, negative_prompt=negative_prompt, generator=generator).images
out_images[0].save("./attack_log/attack_sample/dreambooth1.png")
'''