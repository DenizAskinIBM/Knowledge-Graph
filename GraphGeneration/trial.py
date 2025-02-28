from llms import llm_granite_vision, llm_granite
print(llm_granite_vision.invoke("https://en.wikipedia.org/wiki/Edward_Norton#/media/File:Ed_Norton_Shankbone_Metropolitan_Opera_2009.jpg").content)
print(llm_granite.invoke("How many letters are in the word 'opportunism'?").content)