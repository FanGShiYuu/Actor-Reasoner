import copy
import random
from params import *
import time
import re
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

class EmbeddingWrapper:
    def __init__(self, model_name_or_path):
        self.model = SentenceTransformer(model_name_or_path)

    def embed_documents(self, documents):
        return [self.model.encode(doc).tolist() for doc in documents]

    def embed_query(self, query):
        return self.model.encode(query).tolist()

class DrivingMemory:
    def __init__(self) -> None:
        # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # for the region can not connect to huggingface
        # model = SentenceTransformer("all-MiniLM-L6-v2")  # first time use, active line25&26, when model are downloaded, comment these line to avoid repeared download
        # model.save("model/all-MiniLM-L6-v2/") 
        model_path = 'model/all-MiniLM-L6-v2/'
        self.embedding = EmbeddingWrapper(model_path)
        self.memory_by_type = {}
        self.load_memories()
        self.embedding.embed_query("warmup")
        print("==========Memory loaded with types: ", list(self.memory_by_type.keys()), "==========")

    def load_memories(self):
        for memory_type in ["normal", "aggressive", "conservative"]:
            self.memory_by_type[memory_type] = Chroma(
                embedding_function=self.embedding,
                persist_directory=f'./db/{Scenario_name}/{memory_type}'
            )
            print(Scenario_name)
            print(f"Loaded memory for {memory_type}, now has {len(self.memory_by_type[memory_type]._collection.get(include=['embeddings'])['embeddings'])} items.")

    def determine_memory_type(self, sce_descrip):
        match = re.search(r'Interaction vehicle driving style:\s*(\w+)', sce_descrip)
        if match:
            return match.group(1)
        return 'normal'

    def extract_numeric_value(self, description):
        match = re.search(r'Conflict info: (.*?);', description)
        if match:
            conflict_info = match.group(1)
            return [float(num) for num in re.findall(r'[-+]?\d*\.?\d+', conflict_info)]
        return []

    def remove_numbers(self, text):
        """Remove numbers from the text."""
        return re.sub(r'\d+', '', text)

    def find_closest_numbers(self, input_numbers, stored_numbers, top_k):
        weights = [10, 1, 1, 1, 1]
        distances = []
        for num in stored_numbers:
            weighted_distance = sum(weights[i] * abs(input_numbers[i] - num[i]) for i in range(len(num)))
            distances.append(weighted_distance)
        indices = np.argsort(distances)[:top_k]
        closest_distance = [distances[index] for index in indices]
        closest_indices = [int(index) for index in indices]
        return closest_indices, closest_distance

    def retrieveMemory(self, query_scenario, top_k):
        """Retrieve the most similar scenarios from memory."""
        query_value = self.extract_numeric_value(query_scenario)
        memory_type = self.determine_memory_type(query_scenario)
        stored_memories = self.memory_by_type[memory_type]._collection.get(include=['documents', 'metadatas'])
        stored_numbers = [self.extract_numeric_value(doc) for doc in stored_memories['documents']]

        closest_indices = self.find_closest_numbers(query_value, stored_numbers, 10)[0]
        query_text = self.remove_numbers(query_scenario)
        try:
            similarity_results = self.memory_by_type[memory_type].similarity_search_with_score(query_text, k=top_k, filter={'id':{'$in':closest_indices}})
            fewshot_results = []
            description_results = []
            for idx in range(0, len(similarity_results)):
                fewshot_results.append(similarity_results[idx][0].metadata)
                description_results.append(similarity_results[idx][0].page_content)
                # print(similarity_results[idx])
            return fewshot_results, description_results
        except RuntimeError as e:
            print(f"RuntimeError during HNSW query: {e}, make it slower")
            return [[{'final_action': 'SLOWER'}]]

    def retrieveMemory_without_Tlayer(self, query_scenario, top_k):
        """Retrieve the most similar scenarios from memory."""
        memory_type = self.determine_memory_type(query_scenario)
        try:
            similarity_results = self.memory_by_type[memory_type].similarity_search_with_score(query_scenario, k=top_k)
            fewshot_results = []
            description_results = []
            for idx in range(0, len(similarity_results)):
                fewshot_results.append(similarity_results[idx][0].metadata)
                description_results.append(similarity_results[idx][0].page_content)
                # print(similarity_results[idx])
            return fewshot_results, description_results
        except RuntimeError as e:
            print(f"RuntimeError during HNSW query: {e}, make it slower")
            return [[{'final_action': 'SLOWER'}]]

    def similar_memory_exist(self, memory, sce_descrip, action, number_similarity_threshold=4, text_similarity_threshold=0.8):
        query_value = self.extract_numeric_value(sce_descrip)
        stored_memories = memory._collection.get(include=['documents', 'metadatas'])
        stored_numbers = [self.extract_numeric_value(doc) for doc in stored_memories['documents']]
        if len(stored_numbers) >= 1:
            indices, distance = self.find_closest_numbers(query_value, stored_numbers, 10)
            satisfied_indices = [indices[index] for index in range(len(distance)) if distance[index] < number_similarity_threshold]
            if len(satisfied_indices) >= 1:
                query_text = self.remove_numbers(sce_descrip)
                memory_without_numbers = Chroma(embedding_function=self.embedding)
                for index in satisfied_indices:
                    pc = self.remove_numbers(stored_memories['documents'][index])
                    id_value = stored_memories['metadatas'][index]['id']
                    action = stored_memories['metadatas'][index]['final_action']
                    memory_without_numbers.add_documents([Document(page_content=pc, metadata={'id': id_value, 'final_action': action})])
                similarity_results = memory_without_numbers.similarity_search_with_score(query_text, k=1, filter={'id': {'$in': satisfied_indices}})
                if similarity_results:
                    most_similar_memory = similarity_results[0]
                    if most_similar_memory[1] < text_similarity_threshold:
                        if most_similar_memory[0].metadata['final_action'] == action:
                            return True
                        else:
                            return False
                    else:
                        return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def addMemory(self, sce_descrip, action):
        """Add a new scenario to memory."""
        memory_type = self.determine_memory_type(sce_descrip)
        memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=f'./db/{Scenario_name}/{memory_type}'
        )
        if self.similar_memory_exist(memory, sce_descrip, action):
            print('+++++similar memory exist in dataset, therefore not insert+++++')
        else:
            stored_memories = memory._collection.get(include=['documents', 'metadatas'])
            id_value = len(stored_memories['documents'])
            doc = Document(page_content=sce_descrip, metadata={'id': id_value, 'final_action': action})
            memory.add_documents([doc])
            print('+++++memory successful add into dataset+++++')




