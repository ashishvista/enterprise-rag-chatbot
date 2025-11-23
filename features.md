

for complete enterprise rag chatbot tell me whether this is good stack or not? 
langgraph 
langchain 
llamaindex 
langfuse 
chatwoot or Assistant-UI ?
qdrant or postgres pgvector
BAAI bge-reranker
llm- gpt oss or qwen 3 with tool calling support
chatwoot / liveperson/ aws connect for human agent flow ?
confluence/Amazon Bedrock Knowledge Bases 
suggest me if i am correct or not


top features used here---


1. postgres pg vector for storing embeddings
2. hnsw indexing with cosine similarity search (fastest index and best matching algorithm for RAG usecase)
3. llamaindex semantic chunking using  SemanticSplitterNodeParser. (embeddinggemma)
4. creation of embedings using one of the two most popular opensource
    embeddinggemma (fast, supports dense + sparse hybrid)
    Nomic-embed-text v1.5 

