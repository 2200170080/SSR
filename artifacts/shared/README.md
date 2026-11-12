# Artifact layout

Active entry: src/run_pipeline.py.
Prompt families: datasets/MultiTQ/prompt/{Qwen,Llama,GPT}/. Each contains gp_prompt.txt and qa_prompt.txt.
Indexes: shared/indexes/.
vLLM generation configuration: shared/generation-config/.
History retains old results and raw records.
*.disabled entries and historical-manifest.json are historical only.
New experiments use qwen2.5-32b-awq/runs/.
