#pip install -U openai
#export OPENAI_API_KEY=""

from openai import OpenAI
import os
client = OpenAI()

## Set the API key and model name
MODEL="gpt-4o-mini"
os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

vFile=client.files.create(
  file=open("dataset.jsonl", "rb"),
  purpose="fine-tune"
)

vJob=client.fine_tuning.jobs.create(
  training_file=vFile.id, 
  model="gpt-4o-mini-2024-07-18"
)

print(client.fine_tuning.jobs.list(limit=10))
print(client.fine_tuning.jobs.retrieve(vJob.id))
print(client.fine_tuning.jobs.list_events(fine_tuning_job_id=vJob.id, limit=10))

completion = client.chat.completions.create(
  model=vJob.fine_tuned_model,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)
print(completion.choices[0].message)