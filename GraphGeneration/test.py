from llms import llm_chat_gpt, llm_granite
from codebase import read
transcripts=read("datasets/When_to_verify_the_identity_of_persons_and_entities—Financial_entities.md")
keywords=llm_chat_gpt.invoke("Your task is to read this text and output the name of the sections, the title of the document and keywords that capture the meaning of the whole document. Print these as a list of strings. Do not print any comments, just the list of strings. Document: "+transcripts.strip()).content
question="A client wants to open a savings account, should I verify their identity?"
print(llm_chat_gpt.invoke("For a QUESTION asked, you will be given a LIST of titles and keywords from a document that contains the answer. Your task is to select one title that most likely contains the answer to the question and/or one keyword that corresponds most to it. Print your answer as a list of string. Do not add any comments, just print out a list of strings. QUESTION: "+question+" LIST: "+keywords).content)