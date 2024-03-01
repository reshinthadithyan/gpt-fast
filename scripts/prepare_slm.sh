pip install -r requirements.txt
git lfs install
mkdir ckpts
cd ckpts
git clone https://huggingface.com/stabilityai/stablelm-2-1_6b
hf_path="ckpts/stablelm-2-1_6b"
cd ../
#Convert
python convert_model.py --checkpoint_dir $hf_path --model_name 

#Quantize
python quantize.py --checkpoint_path $hf_path

#Generate
python generate.py --compile --checkpoint_path $hf_path/model.pth --max_new_tokens 100 --prompt "import pandas as" --temperature 0.2