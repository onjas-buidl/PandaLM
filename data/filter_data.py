import json, pandas as pd

with open('data/testset-v1.json', 'r') as f:
    test_cases = json.load(f)

df = pd.DataFrame(test_cases)

df.motivation_app.value_counts()

# filter df by motivation_app in a list of values
df2 = df[df.motivation_app.isin([
    'Wolfram alpha', 'Messenger', 'Grammarly', 'https://cohere.ai/', 'Google Meet'])]
# cohere -> summary
# wolfram -> math
# grammarly -> grammar
# messenger -> chat
# google meet -> meeting summary


# export df2 to json format and save it to data/testset-filtered-141.json
df2.to_json('data/testset-filtered-141.json', orient='records')

#%%
################## analyzing new, filtered, results
with open('data/testset-filtered-141-results.json', 'r') as f:
    test_cases = json.load(f)

df = pd.DataFrame(test_cases)

# mean of PandaLM_result by value of motivation_app
df.groupby('motivation_app').PandaLM_result.mean()
df.groupby('motivation_app').PandaLM_result.var()

# see what's the percentage of 1 for PandaLM_result for each motivation_app group
df.groupby('motivation_app').PandaLM_result.value_counts(normalize=True)

