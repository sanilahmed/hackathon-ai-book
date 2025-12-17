import google.genai as genai

# Initialize the client
client = genai.Client(
    api_key="YOUR_API_KEY"
)

# For embeddings, you would typically use a model that supports embeddings
# The exact API might be different, let me check the models available
response = client.models.list()
for model in response:
    print(model.name)
    if 'embedding' in model.name.lower():
        print(f"Found embedding model: {model.name}")