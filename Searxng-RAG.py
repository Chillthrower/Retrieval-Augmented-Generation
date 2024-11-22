from langchain_community.llms import Ollama
from langchain_community.utilities import SearxSearchWrapper

# Initialize the model
llm = Ollama(model="llama3.1:8b-instruct-q8_0", base_url="https://405d-34-147-121-171.ngrok-free.app")
s = SearxSearchWrapper(searx_host="http://localhost:8888")


question = "Which is the best indian stock to buy?"
context = s.run(question, engines=["brave"])

# Combine the context and question
prompt = f"Context: {context}\nQuestion: {question}"

# Invoke the model with the context and question
response = llm.invoke(prompt)

# Print the response
print(response)


# docker pull searxng/searxng
# docker run -d -p 8888:8080 searxng/searxng
# docker ps
# docker exec -it <container_id_or_name> sh
# vi /etc/searxng/settings.yml
# docker restart <container_id_or_name>

