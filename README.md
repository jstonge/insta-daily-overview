## Notes on using Gemma 4

- I decided to use vllm for serving, working off from [that guide](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#pip-amd-rocm-mi300x-mi325x-mi350x-mi355x)
- The plan is to try `google/gemma-4-31B-it` for image classification.
- Lets try using `vllm serve`, then ping the API from a jupyter notebook
- Didn't work, dependency hell
- lets do [that](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#offline-inference) instead
- dependencies still give trouble, we do the following
```
# we're on h2xnode05
# we loaded the following module
# module load miniforge/25.11.0-py3.12
# module load cuda/12.9.1
uv venv --python python3.12
source .venv/bin/activate
uv pip install transformers==5.5.0 torch torchvision --extra-index-url https://download.pytorch.org/whl/cu129
uv pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu129 --index-strategy unsafe-best-match
uv pip install --upgrade transformers
uv pip install ipykernel jupyter matplotlib openai pillow label-studio-sdk
```
- we're using `uv pip` here because `uv add` is annoying for using nightly. 
- we still had some trouble with the `deep_gem` dependency, we're now following their github repo, which requires us to install [from source](https://github.com/deepseek-ai/DeepGEMM):
```
uv pip install pip # their install script is using pip
cd
# Submodule must be cloned
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM
./install.sh
```
- `deep_gem` is now working, but when stating the jupyter server it is important to make sure we are on the right cuda module, aka `module load cuda/12.9.1`
- we're specifying the vision resolution, following [this section](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html#dynamic-vision-resolution)
- it's finally working, hurray! 
