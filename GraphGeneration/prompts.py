from langchain_core.prompts import PromptTemplate

prompt_graph_generation_llama = PromptTemplate.from_template('''<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a text to knowledge graph translator. Your task is to look at some text and translate it to a knowledge graph.
                                    Here are examples of text to Knowledge Graph format translation:
                                    Text: 'OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.'
                                    Knowledge Graph: GraphDocument(nodes=[Node(id='Osfi', type='Organization', properties={{}}), Node(id='Model Risk Management Framework', type='Framework', properties={{}})], relationships=[Relationship(source=Node(id='Osfi', type='Organization', properties={{}}), target=Node(id='Model Risk Management Framework', type='Framework', properties={{}}), type='OUTLINES', properties={{}})], source=Document(metadata={{}}, page_content='OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.'))
                                    Text: 'Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
                                    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
                                    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.'
                                    Knowledge Graph: GraphDocument(nodes=[Node(id='Marie Curie', type='Person', properties={{}}), Node(id='Pierre Curie', type='Person', properties={{}}), Node(id='Nobel Prize', type='Award', properties={{}}), Node(id='University Of Paris', type='Organization', properties={{}})], relationships=[Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='1867', type='Date', properties={{}}), type='BORN', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Polish', type='Nationality', properties={{}}), type='NATIONALITY', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='French', type='Nationality', properties={{}}), type='NATIONALITY', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Radioactivity', type='Field', properties={{}}), type='RESEARCH', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Nobel Prize', type='Award', properties={{}}), type='WINNER', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='University Of Paris', type='Organization', properties={{}}), type='PROFESSOR', properties={{}}), Relationship(source=Node(id='Pierre Curie', type='Person', properties={{}}), target=Node(id='Nobel Prize', type='Award', properties={{}}), type='WINNER', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Pierre Curie', type='Person', properties={{}}), type='SPOUSE', properties={{}})], source=Document(metadata={{}}, page_content='Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.\nShe was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.\nHer husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.\nShe was, in 1906, the first woman to become a professor at the University of Paris.'))
                                    She was, in 1906, the first woman to become a professor at the University of Paris.
                                    Do not print anything other than the graphs. Do not print any comments.
                                    <|eot_id|><|start_header_id|>user<|end_header_id|>   
                                    Text: {input}
                                    Knowledge Graph:
                                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>''')

prompt_graph_correction_llama = PromptTemplate.from_template('''<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a knowledge graph analyzer. Your task is to fix formatting errors in the Knowledge Graph provided to you. You will be given two examples of text and their translation to the Knowledge Graph format for you to learn the Knowledge Graph formatting.
                                    Then you will be given a knowledge graph to fix. If there are any unmatched parantheses or missing fields, fix them and print the fixed knowledge graph. If the knowledge graph looks good, just print it back.
                                    Here are examples of text to Knowledge Graph format translation:
                                    Text: 'OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.'
                                    Knowledge Graph: GraphDocument(nodes=[Node(id='Osfi', type='Organization', properties={{}}), Node(id='Model Risk Management Framework', type='Framework', properties={{}})], relationships=[Relationship(source=Node(id='Osfi', type='Organization', properties={{}}), target=Node(id='Model Risk Management Framework', type='Framework', properties={{}}), type='OUTLINES', properties={{}})], source=Document(metadata={{}}, page_content='OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.'))
                                    Text: 'Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
                                    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
                                    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.'
                                    Knowledge Graph: GraphDocument(nodes=[Node(id='Marie Curie', type='Person', properties={{}}), Node(id='Pierre Curie', type='Person', properties={{}}), Node(id='Nobel Prize', type='Award', properties={{}}), Node(id='University Of Paris', type='Organization', properties={{}})], relationships=[Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='1867', type='Date', properties={{}}), type='BORN', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Polish', type='Nationality', properties={{}}), type='NATIONALITY', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='French', type='Nationality', properties={{}}), type='NATIONALITY', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Radioactivity', type='Field', properties={{}}), type='RESEARCH', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Nobel Prize', type='Award', properties={{}}), type='WINNER', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='University Of Paris', type='Organization', properties={{}}), type='PROFESSOR', properties={{}}), Relationship(source=Node(id='Pierre Curie', type='Person', properties={{}}), target=Node(id='Nobel Prize', type='Award', properties={{}}), type='WINNER', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Pierre Curie', type='Person', properties={{}}), type='SPOUSE', properties={{}})], source=Document(metadata={{}}, page_content='Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.\nShe was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.\nHer husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.\nShe was, in 1906, the first woman to become a professor at the University of Paris.'))
                                    She was, in 1906, the first woman to become a professor at the University of Paris.
                                    Do not print anything other than the graphs. Do not print any comments.
                                    <|eot_id|><|start_header_id|>user<|end_header_id|>   
                                    Knowledge Graph: {knowledge_graph}
                                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>''')


prompt_chunking_llama = PromptTemplate.from_template(template='''
<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an agentic chunker. You will be provided with content.
Decompose the content into clear and simple propositions, ensuring they are interpretable out of context.
Follow these steps:
1. Split compound sentences into simple sentences, maintaining original phrasing as much as possible.
2. For named entities with additional descriptive info, separate that info into its own proposition.
3. Decontextualize each proposition by adding necessary modifiers and replacing pronouns ("it", "he", "she", "they", "this") with the full name/entity they refer to.
4. Present the results strictly as a list of strings where each string is a chunk.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Here is the content: {input}

Strictly follow the instructions and output ONLY in the desired list format.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''
)

prompt_graph_review_llama = PromptTemplate.from_template('''<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a reviewer. Your task is to look at some text and its translation to a knowledge graph.
                                               Review the knowledge graph: if any nodes and relationships are missing, print a new knowledge graph with the new 
                                               information added; else print DONE
                                    Do not include any comments, just print the new knowledge graph, or DONE
                                    <|eot_id|><|start_header_id|>user<|end_header_id|>   
                                    Text: {input}
                                    Knowledge Graph: {knowledge_graph} <|eot_id|><|start_header_id|>assistant<|end_header_id|>''')


prompt_multiple_graph_query_llama = PromptTemplate.from_template('''<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a question answerer. Your task is to look at a list of Neo4J knowledge graphs and answer questions about them. The knowledge graphs are of transcripts of conversations between a client and custommer support.
                                    Your task is to answer a question in natural language based on the knowledge graphs of these transcripts.
                                    Here is an example of a text to Knowledge Graph format translation:
                                    Text: OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.
                                    Knowledge Graph: GraphDocument(nodes=[Node(id='Osfi', type='Organization', properties={{}}), Node(id='Model Risk Management Framework', type='Framework', properties={{}})], relationships=[Relationship(source=Node(id='Osfi', type='Organization', properties={{}}), target=Node(id='Model Risk Management Framework', type='Framework', properties={{}}), type='OUTLINES', properties={{}})], source=Document(metadata={{}}, page_content='OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.'))
                                    Do not print any comments, just answer the questions.
                                    <|eot_id|><|start_header_id|>user<|end_header_id|>   
                                    Question: {question}
                                    List of Knowledge Graphs: {knowledge_graphs} 
                                    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>''')

prompt_graph_generation_chat_gpt = PromptTemplate.from_template('''You are a text to knowledge graph translator. Your task is to look at some text and translate it to a knowledge graph.
                                    Here is an example of a text to Knowledge Graph format translation:
                                    Text: OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.
                                    Knowledge Graph: GraphDocument(nodes=[Node(id='Osfi', type='Organization', properties={{}}), Node(id='Model Risk Management Framework', type='Framework', properties={{}})], relationships=[Relationship(source=Node(id='Osfi', type='Organization', properties={{}}), target=Node(id='Model Risk Management Framework', type='Framework', properties={{}}), type='OUTLINES', properties={{}})], source=Document(metadata={{}}, page_content='OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.'))
                                    Do not print anything other than the graphs. Do not print any comments.
                                    Text: {input}
                                    Knowledge Graph: ''')

prompt_chunking_chat_gpt = PromptTemplate.from_template(template='''
Decompose the content into clear and simple propositions, ensuring they are interpretable out of context.
Follow these steps:
1. Split compound sentences into simple sentences, maintaining original phrasing as much as possible.
2. For named entities with additional descriptive info, separate that info into its own proposition.
3. Decontextualize each proposition by adding necessary modifiers and replacing pronouns ("it", "he", "she", "they", "this") with the full name/entity they refer to.
4. Present the results strictly as a list of strings where each string is a chunk.

Here is the content: {input}

Strictly follow the instructions and output each chunk in the desired list format.
'''
)

prompt_graph_review_chat_gpt = PromptTemplate.from_template('''You are a reviewer. Your task is to look at some text and its translation to a knowledge graph.
                                               Review the knowledge graph: if any nodes and relationships are missing, print a new knowledge graph with the new 
                                               information added; else print DONE
                                    Do not include any comments, just print the new knowledge graph, or DONE
                                    Text: {input}
                                    Knowledge Graph: {knowledge_graph}''')

prompt_graph_grading_chat_gpt = PromptTemplate.from_template('''You are a reviewer knowledge graph reviewer. Your task is to look at some text and its two translations to Neo4J knowledge graph schemas.
                                                            Your task is to select the knowledge graph schema that captures the most information from the text.
                                                            If the first graph is better, print ONE; else print TWO. Then print your explanation. E.g. "ONE. This graph is better because..."
                                               Text: {text}
                                               Knowledge Graph ONE: {knowledge_graph_1}
                                               Knowledge Graph TWO: {knowledge_graph_2} ''')

prompt_graph_query_chat_gpt = PromptTemplate.from_template('''You are a question answerer. Your task is to look at a Neo4J knowledge graph and answer questions about it.
                                    Do not include any comments, just print the answer.
                                    Question: {question}
                                    Knowledge Graph: {knowledge_graph} ''')

prompt_multiple_graph_query_chat_gpt = PromptTemplate.from_template('''You are a question answerer. Your task is to look at a list of Neo4J knowledge graphs and answer questions about them. The knowledge graphs are of transcripts of conversations between a client and custommer support.
                                    Your task is to answer a question based on the knowledge graphs of these transcripts. Do not print any comments, just answer the questions.
                                    Question: {question}
                                    List of Knowledge Graphs: {knowledge_graphs} ''')

prompt_graph_generation_mistral = PromptTemplate.from_template('''[INST] You are a text to knowledge graph translator. Your task is to look at some text and translate it to a knowledge graph.
                                    Here are examples of text to Knowledge Graph format translation:
                                    Text: 'OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.'
                                    Knowledge Graph: GraphDocument(nodes=[Node(id='Osfi', type='Organization', properties={{}}), Node(id='Model Risk Management Framework', type='Framework', properties={{}})], relationships=[Relationship(source=Node(id='Osfi', type='Organization', properties={{}}), target=Node(id='Model Risk Management Framework', type='Framework', properties={{}}), type='OUTLINES', properties={{}})], source=Document(metadata={{}}, page_content='OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.'))
                                    Text: 'Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
                                    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
                                    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.'
                                    Knowledge Graph: GraphDocument(nodes=[Node(id='Marie Curie', type='Person', properties={{}}), Node(id='Pierre Curie', type='Person', properties={{}}), Node(id='Nobel Prize', type='Award', properties={{}}), Node(id='University Of Paris', type='Organization', properties={{}})], relationships=[Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='1867', type='Date', properties={{}}), type='BORN', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Polish', type='Nationality', properties={{}}), type='NATIONALITY', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='French', type='Nationality', properties={{}}), type='NATIONALITY', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Radioactivity', type='Field', properties={{}}), type='RESEARCH', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Nobel Prize', type='Award', properties={{}}), type='WINNER', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='University Of Paris', type='Organization', properties={{}}), type='PROFESSOR', properties={{}}), Relationship(source=Node(id='Pierre Curie', type='Person', properties={{}}), target=Node(id='Nobel Prize', type='Award', properties={{}}), type='WINNER', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Pierre Curie', type='Person', properties={{}}), type='SPOUSE', properties={{}})], source=Document(metadata={{}}, page_content='Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.\nShe was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.\nHer husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.\nShe was, in 1906, the first woman to become a professor at the University of Paris.'))
                                    She was, in 1906, the first woman to become a professor at the University of Paris.
                                    Do not print anything other than the graphs. Do not print any comments.[/INST] 
                                    Text: {input}
                                    Knowledge Graph: ''')

prompt_chunking_mistral = PromptTemplate.from_template(template='''
[INST] Decompose the content into clear and simple propositions, ensuring they are interpretable out of context.
Follow these steps:
1. Split compound sentences into simple sentences, maintaining original phrasing as much as possible.
2. For named entities with additional descriptive info, separate that info into its own proposition.
3. Decontextualize each proposition by adding necessary modifiers and replacing pronouns ("it", "he", "she", "they", "this") with the full name/entity they refer to.
4. Present the results strictly as a list of strings where each string is a chunk. [/INST]

Here is the content: {input}

Strictly follow the instructions and output each chunk in the desired list format.
''')

prompt_graph_review_mistral = PromptTemplate.from_template('''[INST] You are a reviewer. Your task is to look at some text and its translation to a knowledge graph.
                                               Review the knowledge graph: if any nodes and relationships are missing, print a new knowledge graph with the new 
                                               information added; else print DONE
                                    Do not include any comments, just print the new knowledge graph, or DONE [/INST]
                                    Text: {input}
                                    Knowledge Graph: {knowledge_graph}''')

prompt_multiple_graph_query_mistral = PromptTemplate.from_template('''[INST] You are a question answerer. Your task is to look at a list of Neo4J knowledge graphs and answer questions about them. The knowledge graphs are of transcripts of conversations between a client and custommer support.
                                    Your task is to answer a question based on the knowledge graphs of these transcripts.
                                    Here are two examples of text to Knowledge Graph format translation for you to learn the knowledge graph format:
                                    Text: 'OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.'
                                    Knowledge Graph: GraphDocument(nodes=[Node(id='Osfi', type='Organization', properties={{}}), Node(id='Model Risk Management Framework', type='Framework', properties={{}})], relationships=[Relationship(source=Node(id='Osfi', type='Organization', properties={{}}), target=Node(id='Model Risk Management Framework', type='Framework', properties={{}}), type='OUTLINES', properties={{}})], source=Document(metadata={{}}, page_content='OSFI outlines its expectations for the establishment of an enterprise-wide model risk management framework at institutions.'))
                                    Text: 'Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
                                    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
                                    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.'
                                    Knowledge Graph: GraphDocument(nodes=[Node(id='Marie Curie', type='Person', properties={{}}), Node(id='Pierre Curie', type='Person', properties={{}}), Node(id='Nobel Prize', type='Award', properties={{}}), Node(id='University Of Paris', type='Organization', properties={{}})], relationships=[Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='1867', type='Date', properties={{}}), type='BORN', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Polish', type='Nationality', properties={{}}), type='NATIONALITY', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='French', type='Nationality', properties={{}}), type='NATIONALITY', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Radioactivity', type='Field', properties={{}}), type='RESEARCH', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Nobel Prize', type='Award', properties={{}}), type='WINNER', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='University Of Paris', type='Organization', properties={{}}), type='PROFESSOR', properties={{}}), Relationship(source=Node(id='Pierre Curie', type='Person', properties={{}}), target=Node(id='Nobel Prize', type='Award', properties={{}}), type='WINNER', properties={{}}), Relationship(source=Node(id='Marie Curie', type='Person', properties={{}}), target=Node(id='Pierre Curie', type='Person', properties={{}}), type='SPOUSE', properties={{}})], source=Document(metadata={{}}, page_content='Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.\nShe was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.\nHer husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.\nShe was, in 1906, the first woman to become a professor at the University of Paris.'))
                                    She was, in 1906, the first woman to become a professor at the University of Paris.                                   Do not print any comments, just answer the questions. [/INST] 
                                    Question: {question}
                                    List of Knowledge Graphs: {knowledge_graphs} ''')

