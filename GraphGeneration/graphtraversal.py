import nltk
import networkx as nx
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import os

# Make sure these are downloaded (run once).
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

class GraphQA:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.stopwords = set(stopwords.words('english'))
        # Track the last named entity we encountered, to do naive pronoun resolution
        self.last_named_entity = None

    def _normalize_tokens(self, words):
        """
        Small helper that lowercases words except we keep track of the
        original text for naive pronoun resolution. E.g., if words[i] is "She"
        and we have a known last_named_entity, we replace it in place.
        """
        for i, w in enumerate(words):
            w_lower = w.lower()
            # Naive pronoun replacement:
            if w_lower in ["she", "her", "he", "him", "his"] and self.last_named_entity:
                words[i] = self.last_named_entity
            else:
                words[i] = w_lower
        return words

    def _extract_svo_triples(self, sentence):
        """
        Naively extract (subject, verb, object) sets from a single sentence.
        Steps:
         1) Tokenize & do naive pronoun resolution.
         2) Tag POS.
         3) Identify first NNP (or NN) as the subject (or continue the previous subject).
         4) For each verb we see, treat all subsequent noun chunks as objects,
            until next verb or sentence ends.
         5) If multiple consecutive objects are found (split by 'and'), link them individually.
        """
        # 1) Tokenize
        tokens = word_tokenize(sentence)
        # Replace pronouns if possible
        tokens = self._normalize_tokens(tokens)
        # 2) Tag POS
        tagged = pos_tag(tokens)

        # We will store partial results here
        subject = None
        triples = []
        current_verb = None

        # Because we want to combine multi-word entities, let's build them on the fly:
        def flush_entity_buffer(buf):
            # Combine everything into a single string
            if not buf:
                return None
            entity_str = " ".join(buf).strip().lower()
            return entity_str

        entity_buffer = []

        def finalize_entity():
            # Turn the buffered tokens into one entity, clear buffer
            nonlocal entity_buffer
            ent = flush_entity_buffer(entity_buffer)
            entity_buffer = []
            return ent

        i = 0
        while i < len(tagged):
            word, pos = tagged[i]
            # If we see a verb, flush any entity buffer as an object or subject
            if pos.startswith("VB"):
                # If we had a buffered entity and no subject yet, that becomes subject
                if subject is None and entity_buffer:
                    subject = finalize_entity()
                    # Store as "last_named_entity" if it appears to be a proper noun phrase
                    if any(t[1].startswith("NNP") for t in tagged if t[0].lower() == subject):
                        self.last_named_entity = subject
                # This verb becomes the "current verb" for subsequent objects
                current_verb = word
                i += 1
                continue

            # If it's punctuation or a stop "the/and/etc." that breaks up an entity, flush the buffer
            if pos in ["CC", "IN", "DT", "TO", "PRP", "PRP$"] or word in [",", ".", "and"]:
                # If we had an entity in the buffer, see if it becomes subject or object
                ent = finalize_entity()
                if ent:
                    # If we have no subject and no verb yet, that is our subject
                    if subject is None and current_verb is None:
                        subject = ent
                        # If the POS is NNP, track last_named_entity
                        if any(t[1].startswith("NNP") for t in tagged if t[0].lower() == ent):
                            self.last_named_entity = ent
                    # Else if we have a verb, treat that entity as an object
                    elif current_verb:
                        # Register triple
                        triples.append((subject, current_verb, ent))
                        # If the entity looks like a named entity, store it
                        if any(t[1].startswith("NNP") for t in tagged if t[0].lower() == ent):
                            self.last_named_entity = ent
                i += 1
                continue

            # If it's a noun (NN, NNP, NNS, NNPS) or an adjective/number that might be part of a compound
            if pos.startswith("NN") or pos.startswith("JJ") or pos.startswith("CD"):
                # Accumulate it into the entity_buffer
                entity_buffer.append(word)
                i += 1
            else:
                # For anything else, just skip or flush if needed
                i += 1

        # End of sentence; flush leftover entity
        ent = finalize_entity()
        if ent:
            if subject is None and current_verb is None:
                subject = ent
                # Possibly update last_named_entity
                self.last_named_entity = ent
            elif current_verb:
                triples.append((subject, current_verb, ent))
                # Possibly update last_named_entity
                self.last_named_entity = ent

        return triples

    def extract_entities_and_relationships(self, paragraph):
        """
        Parse the paragraph sentence by sentence, do naive pronoun resolution, 
        extract SVO triples, and build edges in the DiGraph.
        """
        sentences = nltk.sent_tokenize(paragraph)
        for sent in sentences:
            triples = self._extract_svo_triples(sent)
            for (subj, verb, obj) in triples:
                # Add an edge for each triple (subject -> object, relation=verb)
                if subj and obj:
                    self.graph.add_edge(subj, obj, relation=verb)

    def answer_question(self, question):
        """
        Very naive approach to handle:
          1) "Who discovered X?"
          2) "What did X win?"
          3) Otherwise, we try normal shortest_path() if we detect 2 entities.

        You can expand to handle more question patterns.
        """
        qtokens = word_tokenize(question.lower())
        # Simple detection
        if qtokens[0] == "who" and "discovered" in qtokens:
            # Then presumably "X" is the last noun
            # e.g. "Who discovered radium?"
            # We'll gather all nodes that have an edge to "radium" with relation=discovered
            # and return them
            for w, pos in pos_tag(qtokens):
                if pos.startswith("NN"):
                    obj_entity = w
                    # Search for any node that discovered obj_entity
                    discoverers = []
                    for node in self.graph.nodes():
                        if self.graph.has_edge(node, obj_entity):
                            if "discover" in self.graph[node][obj_entity]["relation"]:
                                discoverers.append(node)
                    if discoverers:
                        return f"{', '.join(discoverers)} discovered {obj_entity}"
            return "I don't know who discovered that."

        if qtokens[0] == "what" and "win" in qtokens:
            # "What did X win?"
            # Let's guess the subject X is the big noun phrase after "did"
            # e.g. "What did Albert Einstein win?"
            # We'll look for an entity named "albert einstein" in the graph,
            # then see all edges from it that have 'win' or 'won' in the relation.
            # We'll collect all the objects it leads to.
            words_tagged = pos_tag(qtokens)
            # Grab the first large NNP chunk after "did"
            # We'll do a quick approach: everything from "did" to "win" ignoring
            # "did" itself and "win" itself for the subject
            subj_tokens = []
            record = False
            for w, pos in words_tagged:
                if w == "did":
                    record = True
                    continue
                if w.startswith("win"):
                    break
                if record:
                    subj_tokens.append(w)
            subject_candidate = " ".join(subj_tokens).strip()

            if subject_candidate in self.graph.nodes():
                # Look for edges subject_candidate -> some_object with relation=win or won
                # or containing 'win'
                results = []
                for target in self.graph[subject_candidate]:
                    rel = self.graph[subject_candidate][target].get("relation", "")
                    if "win" in rel or "won" in rel:
                        results.append(target)
                if results:
                    return f"{subject_candidate} won {', '.join(results)}"
                else:
                    return "I don't see anything that they won in the graph."
            else:
                return "Entities not found in the knowledge graph."

        # Fallback: see if we can do normal entity-based shortest_path
        # Let's find all noun phrases in the question to see if we have at least 2
        question_entities = []
        for w, pos in pos_tag(qtokens):
            if pos.startswith("NN"):
                question_entities.append(w)

        if len(question_entities) < 2:
            return "Insufficient entities in the question."

        start, end = question_entities[0], question_entities[-1]
        if start not in self.graph or end not in self.graph:
            return "Entities not found in the knowledge graph."
        try:
            path = nx.shortest_path(self.graph, start, end)
            edges = []
            for i in range(len(path)-1):
                rel = self.graph[path[i]][path[i+1]]["relation"]
                edges.append(f"{path[i]} ({rel}) {path[i+1]}")
            return " -> ".join(edges)
        except nx.NetworkXNoPath:
            return "No connection found."

# -------------------------------
# Example Usage
# -------------------------------
paragraph = """
Napoleon Bonaparte[b] (born Napoleone Buonaparte;[1][c] 15 August 1769 – 5 May 1821), later known by his regnal name Napoleon I, was a French general and statesman who rose to prominence during the French Revolution and led a series of military campaigns across Europe during the French Revolutionary and Napoleonic Wars from 1796 to 1815. He led the French Republic as First Consul from 1799 to 1804, then ruled the French Empire as Emperor of the French from 1804 to 1814, and briefly again in 1815.
"""

qa_system = GraphQA()
qa_system.extract_entities_and_relationships(paragraph)

questions = [
    "Where was Napoleon from?"
]

for question in questions:
    print(f"Q: {question}")
    print(f"A: {qa_system.answer_question(question)}\n")