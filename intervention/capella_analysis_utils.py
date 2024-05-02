from sentence_transformers import SentenceTransformer, util
import torch 
from ast import literal_eval
# from test_reasoning_moon import load_model, evaluate#
from reasoning import load_model, evaluate
import os 
import capellambse
import json
from collections import deque
from tqdm import tqdm

def get_neighbors_component_bfs(root_component, visited=None, max_triples=None):
    queue = deque([root_component])
    triples = []
    if visited is None:
        visited = set()
    
    while queue:
        
        if max_triples and len(triples) >= max_triples:
            break  # Stop if the maximum number of triples has been reached

        component = queue.popleft()
        #print(component.name)
        ini_id = component.uuid

        if ini_id in visited:
            continue

        visited.add(ini_id)

        # Process direct component neighbors
        for comp in component.components:
            triples.append((comp.xtype, ini_id, comp.uuid, component.name, "directly_contains", comp.name))
            queue.append(comp)
        
        # Add parent to the queue
        try:
            triples.append((component.xtype, ini_id, component.uuid, component.name, "is_contained_by", component.owner.name))
            queue.append(component.owner)
        except AttributeError: 
            print(component.name, "has no owner")

        # Add constraints
        for const in component.constraints:
            triples.append((const.xtype, ini_id, const.uuid, component.name, "constraint", const.name))

        # Add property values
        for prop in component.property_values:
            triples.append((prop.xtype, ini_id, prop.uuid, component.name, "has", prop.name +" " + str(prop.value)))

        try: 
            # Add component exchanges
            for exch in component.related_exchanges:
                if exch.target.owner.name != component.name: 
                    target = exch.target
                else: 
                    target = exch.source
                triples.append((target.owner.xtype, ini_id, target.owner.uuid, component.name, exch.name, target.owner.name))
                queue.append(target.owner)
        except AttributeError:
            print("Error Exchanges")
            print(component.name)
            pass

        try:    
            # Process functions and related exchanges
            for func in component.allocated_functions:
                triples.append((func.xtype, component.uuid, func.uuid, component.name, "directly_performs", func.name))
                process_function_exchanges(func, triples, component, queue, visited)

        except AttributeError:
            print("Error Functions")
            print(component.name)
            pass

        #print(triples)
    return triples

def process_function_exchanges(func, triples, component, queue, visited):
    # Process inputs and outputs of the function
    for input in func.inputs:
        if len(input.exchanges) == 1:
            source = input.exchanges[0]
            triples.append((source.xtype, func.uuid, source.source.owner.uuid, func.name, source.name, source.source.owner.name))
            add_parent_triple(source.source.owner, triples, queue, visited)

    for output in func.outputs:
        if len(output.exchanges) == 1:
            target = output.exchanges[0]
            triples.append((target.xtype, func.uuid, target.target.owner.uuid, func.name, target.name, target.target.owner.name))
            add_parent_triple(target.target.owner, triples, queue, visited)

def add_parent_triple(parent, triples, queue, visited):
    # Helper function to add a triple for the parent component and add it to the queue
    if parent is not None and parent.uuid not in visited:
        triples.append((parent.xtype, parent.uuid, parent.owner.uuid, parent.name, "is_performed_by", parent.owner.name))
        queue.append(parent.owner)

def semantic_search(query, model_capella, model_emb, top_k=10, verbose = False):

    #query = req

    names = [comp.name for comp in model_capella.la.all_components]

    query_embedding = model_emb.encode(query, convert_to_tensor=True)

    embeds = model_emb.encode(names, convert_to_tensor=True)

    top_k = 10

    cos_scores = util.cos_sim(query_embedding, embeds)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    if verbose: 
       
        print("======================")
        print("Input query:", query)
        print(f"\nTop most similar entries in Knowledge Graph:")

        #iids = [iids[idx] for idx in top_results.indices.tolicst()]
        #names= get_names([iids[idx] for idx in top_results.indices.tolist()], session) # inputs['artifact']['type'],
        # scores = [top_results[idx] for idx in top_results.indices.tolist()]
        # print(names)
        for score, idx in zip(top_results[0], top_results[1]):
        #for score, name in zip(top_results[0], names):
            # print(iids[idx], "(Score: {:.4f})".format(score))
            print(names[idx], "(Score: {:.4f})".format(score))

    entities = [names[idx] for idx in top_results.indices.tolist()]
    return entities 

def parse_output(output, prompt_reason, tokenizer):

    output =output[len(tokenizer.decode(tokenizer.encode(prompt_reason))):]
    return output

def clean_text(text):
    text = text.split("<|end of turn|>")[0]
    text = text.strip()
    return text

def rerank(model, tokenizer, prompt, extra_args): 

    output, complete_output = evaluate(prompt, model = model, tokenizer= tokenizer,  device = "cuda", **extra_args)
    parsed_output = parse_output(output, prompt, tokenizer)
    parsed_output = clean_text(parsed_output)
    parsed_output = literal_eval(parsed_output)

    assert type(parsed_output) == list
    return parsed_output

import torch 
def main(inputs):
    # Example usage:
    # Assuming 'root_component' is your starting component
    # and 'max_triples' is the maximum number of triples you want to insert
    #results = get_neighbors_component_bfs(root_component, max_triples=50)
    #root_component = model_capella.la.all_components.by_name(names[top_results[1][0]])
    script_dir = os.path.abspath(os.path.dirname(__file__))

    reqs = inputs['reqs']
    path_to_model = "./capella/moonbase/SGAC_Moon_Base/IP_New.aird"
    model_capella = capellambse.MelodyModel(path_to_model)

    model_emb = SentenceTransformer('all-mpnet-base-v2')

    
    inputs['model'] = "openchat/openchat_3.5"
    tokenizer, model = load_model(inputs['model'])
    for req in tqdm(reqs): 
        query = req['requirement']

        entities = semantic_search(query, model_capella, model_emb, top_k= 10)
        if RERANK: 
            prompt = f"""GPT4 Correct User: Based on the requirement and given entities what are the most relevant entities? Return only a Python list object.

            Requirement: The settlement shall enable regular crew and cargo access to the lunar surface (and vice versa) via airlock or pressurized rover.
            Entities: ['Settlement Peripherals & Payloads', 'Habitation', 'Pressurized Rover', 'Airlock', 'Lander', 'Surface Robotics', 'Uncrewed Surface Robotics', 'Structure, Shielding, & Storage', 'Thermal Control', 'Waste Management']
            <|end of turn|>GPT4 Correct Assistant: ['Airlock', 'Pressurized Rover']
            <|end of turn|>GPT4 Correct User: Requirement: {query}
            Entities: {entities}            
            <|end of turn|>GPT4 Correct Assistant:
            """
            MAX_NEW_TOKENS = 128
            extra_args = {"max_new_tokens":MAX_NEW_TOKENS,
            #  "min_new_tokens": 256, 
                "do_sample":False,#True, 
                "num_beams" : 1, 
                "num_return_sequences" : 1, 
                #"no_repeat_ngram_size": 12, 
                "temperature": 0.5, 
                "top_p": 0.95,
            #  "begin_suppress_tokens": [2], 
                }

            try: 
                entities = rerank(model, tokenizer, prompt, extra_args)
                entities = entities[0:1]
                print(len(entities))
            except:
                print("Error")
                print(req['requirement'])
                
                #print(entities)
                entities = entities[0:1]

        else: 
            entities = entities[0:1]

        results = []
        MAX_TRIPLES = 60
        for entity in entities:
            max_triples = MAX_TRIPLES // len(entities)
            print(max_triples)
            try: 
                root_component = model_capella.la.all_components.by_name(entity)
            except: 
                print(entity)
                entities.remove(entity)
                continue
            temp = get_neighbors_component_bfs(root_component, max_triples= max_triples)

            for res in temp[0:max_triples]:
                results.append(res)

        definition = ""
        for triples in results: 
            definition += "|" + triples[3] +"|" +" " + "|" + triples[4] +"|" +" " +"|"+ triples[5] +"|" + "\n"

        #req['definition_no_rerank'] = definition
        req['definition'] = definition

        # requirement = req
        # prompt = f"""GPT4 Correct User: Based on the requirement what is the most relevant entities? Return only a Python list object.
        # Entities are: {entities}
        # Requirement: {requirement}
        # <|end of turn|>GPT4 Correct Assistant: Entity:
        # """

        # entities = rerank(model, tokenizer, prompt, extra_args)

    with open(os.path.join(script_dir, inputs['output_path']), "w") as outfile:
        json.dump(reqs, outfile, indent=4)
    
    print("done")
    #print(definition) 

if __name__ == "__main__":
    
    RERANK = True
    with open("moonbase_reqs.json", "r") as file:
        reqs = json.load(file)
    inputs = { "reqs": reqs, 
              "output_path": "moonbase_reqs_2.json" }
    
    main(inputs)