import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, required=True, help="Path to the original model")
parser.add_argument("--output-path", type=str, required=True, help="Path to save the converted model")

args = parser.parse_args()

model = torch.load(f"{args.model_path}/pytorch_model.bin")

for name in list(model.keys()):
    if ".q_proj.weight" in name:
        print(f"Processing {name}")
        model[name] = model[name].view(2, 16, 128, -1).transpose(0, 1).reshape(4096, -1).contiguous()
    if ".o_proj.weight" in name:
        print(f"Processing {name}")
        model[name] = model[name].view(-1, 2, 16, 128).transpose(1, 2).reshape(-1, 4096).contiguous()

torch.save(model, f"{args.output_path}/pytorch_model.bin")