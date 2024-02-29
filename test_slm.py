from model import Transformer, ModelArgs
# from transformers import AutoModelForCausalLM


# model = AutoModelForCausalLM.from_pretrained("/weka/home-reshinth/ckpts/stablelm-2-1_6b")
# print(model)
# print(model.model.layers[0].self_attn.q_proj.weight.size())
args = ModelArgs.from_name("StableLM-1_6B") 
model = Transformer(args)
#/weka/home-reshinth/ckpts/stablelm-2-1_6b
print(model)
print(model.layers[0].attention.wqkv.bias.size())