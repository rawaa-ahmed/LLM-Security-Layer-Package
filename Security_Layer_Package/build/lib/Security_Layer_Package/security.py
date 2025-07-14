# import torch

import models
###############################################################################################
# Importing stock ml libraries
from flask import Flask, request, jsonify
import warnings
warnings.simplefilter('ignore')
# import pandas as pd
import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModel
# import logging
# import os
import threading
import time
# import concurrent.futures
import ctypes

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# Security_Layer = Flask(__name__)
###############################################################################################
class TerminableThread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.execution_time = None
        self.daemon = True

    def run(self):
        start_time = time.time()
        self.result = self.func(*self.args, **self.kwargs)
        end_time = time.time()
        self.execution_time = round(end_time - start_time, 3)

    def get_id(self):
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def terminate(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            #print('Exception raise failure')

class FunctionExecutor:
    def __init__(self, timeout):
        self.timeout = timeout
        self.completed_results = []
        self.threads = []

    def execute(self, functions):
        start_time = time.time()

        for func, args, kwargs in functions:
            thread = TerminableThread(func, *args, **kwargs)
            self.threads.append(thread)
            thread.start()

        while self.threads:
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout:
                #print("Timeout reached. Terminating remaining functions.")
                for thread in self.threads:
                    thread.terminate()
                break

            for thread in list(self.threads):
                if not thread.is_alive():
                    self.threads.remove(thread)
                    if thread.result is not None:
                        result_with_time = {
                            'result': thread.result,
                            'execution_time': thread.execution_time
                        }
                        self.completed_results.append(result_with_time)
                        print(f"Completed: {thread.result}, Time Taken: {thread.execution_time} seconds")

            time.sleep(0.1)  # Small sleep to prevent busy waiting

        #print("All done.")
        return self.completed_results
###############################################################################################
###############################################################################################
###############################################################################################
def Input_Judgement_layer(results,threshold=0.7):
    # Initialize default values
    prompt_injection_score = None
    cipher_free = None
    mal_safe_score = None
    relevance_score = None
    topic = None
    offencive_content_score = None

    # Parse the result data
    for item in results:
        result = item.get('result', {})
        if 'Prompt_Injection_Detector' in result:
            prompt_injection_score = result['Prompt_Injection_Detector'].get('score', None)
        if 'Offencive_Content_Detector' in result:
            offencive_content_score = result['Offencive_Content_Detector'].get('score', None)
        if 'Cipher_Free' in result:
            cipher_free = result['Cipher_Free']
        if 'input_One_Agent' in result:
            mal_safe_score = result['input_One_Agent'].get('malicious_safe_score', None)
            relevance_score = result['input_One_Agent'].get('relevance_score', None)
            topic = result['input_One_Agent'].get('topic_classification', "Unknown")
    
    Fail_Responce = {"Response" : "Fail"}
    Pass_Responce = {"Response" : "Pass"}

    # Decision based on Prompt_Injection_Detector
    if prompt_injection_score is not None and prompt_injection_score >= threshold:
        return Fail_Responce
    
    # Decision based on Offencive_Content_Detector
    #if offencive_content_score is not None and offencive_content_score >= threshold:
    #    return Fail_Responce

    # Decision based on Cipher_Free
    if cipher_free is not None and not cipher_free:
        return Fail_Responce
    
    # Decision based on input_One_Agent
    #if mal_safe_score is not None and mal_safe_score >= 0.3:
    #    return Fail_ResponceS
    
    if len(results) == 0: 
        return "the layer couldn't analysis the input within the time"
    # Default pass if no failing conditions are met
    return Pass_Responce
############################################################################################
############################################################################################
############################################################################################
def Prompt_Injection_Detector(prompt,language,threshold):
    Evaluation = models.prompt_injection_detection(prompt, language, threshold)
    return {"Prompt_Injection_Detector": {"score" : Evaluation["score"],"class":Evaluation["PID_output"]}}
############################################################################################ 
# def Cipher_Detector(prompt: str):
#     return {"Cipher_Free": cipher_detection(prompt)}
# ############################################################################################
# def Input_evaluator(prompt:str):
#     return {"input_One_Agent": input_one_agent(prompt)}
############################################################################################ 
def Offensive_Content_Detection(prompt,language,threshold):
    Evaluation = models.offencive_content_detection(prompt, language, threshold)
    return {"Offencive_Content_Detector": {"score" : Evaluation["score"],"class":Evaluation["PID_output"]}}
############################################################################################ 
# def Offencive_Content_Detection_Agent(prompt):
#     Evaluation = Objectionable_Detection_Agent(prompt)
#     return Evaluation
############################################################################################
############################################################################################ 
############################################################################################ 
# def Output_evaluator_back_translation(llm_responce:str,llm_role: str):
#     return {"Output_back_translation": back_translation(llm_responce,llm_role)}
# ############################################################################################ 
# def Output_evaluator_one_agent(llm_responce:str,llm_role: str):
#     return {"Output_One_Agent": output_one_agent(llm_responce,llm_role)}
# ############################################################################################
# def Output_evaluator_self_critique(llm_responce:str,llm_role: str):
#     return {"Output_self_critique": self_critique(llm_responce,llm_role)}
# ############################################################################################
# def Output_evaluator_One_Agent(llm_role: str ,llm_responce:str, query:str):
#     extractor = Output_One_Agent(token="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")
#     final_answer = extractor.generate_response(llm_role,llm_responce,query)
#     return {"Output_One_Agent": final_answer}
############################################################################################
############################################################################################
# @Security_Layer.route('/Input_Filtering_Layer', methods=['POST'])
# def Input_Defending_Layer():
#     data = request.get_json()
#     print(data)
#     if not 'prompt' in data:
#         return jsonify({"message": 'Please enter valid input'}), 500
    
#     # handling language value
#     if "language" in data and not ((data['language'] == 'Arabic') or (data['language']  == 'English')):   
#         return jsonify({"message": "Please enter valid language [Arabic/English]"}), 500 
    
#     # handling threshold value
#     if "threshold" in data and not (isinstance(data['threshold'], float) and 0 <= data['threshold'] <= 1):
#         return jsonify({"message": "Please enter valid threshold [0:1]"}), 500 
#     # handling llm role
#     #if not "LLM_ROLE" in data:
#     #    return jsonify({"message": "Please enter the llm role"}), 500
    
#     print(data['prompt'])
    
#     # Extract necessary data (language, threshold, etc.)
#     prompt = data["prompt"]
#     language = data.get('language', None)
#     threshold = data.get('threshold', 0.7)
#     timeout_period = data.get('timeout_period', 10)
#     def process_request(prompt,language,threshold,timeout_period):
#         try:
#             executor = FunctionExecutor(timeout_period)
#             functions = [
#                 (Prompt_Injection_Detector, (prompt,language,threshold), {}),  
#                 (Cipher_Detector, (prompt,), {})
#                 #(Input_evaluator, (prompt,), {})
#                 #(Offencive_Content_Detection, (prompt,language,threshold), {})
#                     ]
#             results = executor.execute(functions)
            
#             if not results:
#                 # If no function completed, return a timeout message
#                 return {"Response": "Fail", "message": "Request timed out"}
#             else:
#                 # Return only the completed results
#                 return results
#         except Exception as e:
#             return {"Response": "Fail", "error": str(e)}
        
#     executor = concurrent.futures.ThreadPoolExecutor()
#     future = executor.submit(process_request, prompt,language,threshold, timeout_period)

#     try:
#         # Wait for the result, letting process_request handle timeout behavior
#         result = future.result()  # Don't pass the timeout here
#         return jsonify(Input_Judgement_layer(result)), 200
#     except Exception as e:
#         return jsonify({"Response": "Fail", "error": str(e)}), 200
    
# ############################################################################################
# """
# @Security_Layer.route('/Output_Filtering_Layer', methods=['POST'])
# def Output_Defending_Layer():
#     data = request.get_json()
#     print(data)
#     if not 'llm_responce' in data:
#         return jsonify({"message": 'Please enter valid llm_responce'}), 500
#     # handling llm role
#     if not "LLM_ROLE" in data:
#         return jsonify({"message": "Please enter the llm role"}), 500
#     if not "query" in data:
#         return jsonify({"message": "Please enter the input query"}), 500
    
#     print(data['llm_responce'])
    
#     if "timeout_period" not in data:
#         data['timeout_period'] = 10
    
#     timeout_period = data['timeout_period']
#     executor = FunctionExecutor(timeout_period)
    
#     functions = [
#         (Output_evaluator_One_Agent, (data["LLM_ROLE"],data["llm_responce"],data["query"]), {})       
#     ]
    
#     results = executor.execute(functions)
#     try : 
#         return jsonify(results[0]['result']["Output_One_Agent"]), 200 
#     except:
#         return jsonify("timeout_reached"), 200     
# """
# ############################################################################################
# """
# @Security_Layer.route('/Objectionable_Detection_Layer', methods=['POST'])
# def Objectionable_Detection_Layer():
#     data = request.get_json()
#     print(data)
#     if not 'prompt' in data:
#         return jsonify({"message": 'Please enter valid input'}), 500
#     print(data['prompt'])
#     return jsonify(Offencive_Content_Detection_Agent(data["prompt"])), 200
# """
# ############################################################################################
# @Security_Layer.route('/Offencive_content_detection_Layer', methods=['POST'])
# def Offencive_content_detection_Layer():
#     data = request.get_json()
#     print(data)
#     if not 'prompt' in data:
#         return jsonify({"message": 'Please enter valid input'}), 500
    
#     # handling language value
#     if "language" in data and not ((data['language'] == 'Arabic') or (data['language']  == 'English')):   
#         return jsonify({"message": "Please enter valid language [Arabic/English]"}), 500 
    
#     # handling threshold value
#     if "threshold" in data and not (isinstance(data['threshold'], float) and 0 <= data['threshold'] <= 1):
#         return jsonify({"message": "Please enter valid threshold [0:1]"}), 500 
#     # handling llm role
#     #if not "LLM_ROLE" in data:
#     #    return jsonify({"message": "Please enter the llm role"}), 500
    
#     print(data['prompt'])
    
#     # Extract necessary data (language, threshold, etc.)
#     prompt = data["prompt"]
#     language = data.get('language', None)
#     threshold = data.get('threshold', 0.7)
#     timeout_period = data.get('timeout_period', 10)
#     def process_request(prompt,language,threshold,timeout_period):
#         try:
#             executor = FunctionExecutor(timeout_period)
#             functions = [
#                             (Offencive_Content_Detection, (prompt,language,threshold), {})  
#                         ]
    

#             results = executor.execute(functions)
            
#             if not results:
#                 # If no function completed, return a timeout message
#                 return {"Response": "Fail", "message": "Request timed out"}
#             else:
#                 # Return only the completed results
#                 return results
#         except Exception as e:
#             return {"Response": "Fail", "error": str(e)}
        
#     executor = concurrent.futures.ThreadPoolExecutor()
#     future = executor.submit(process_request, prompt,language,threshold, timeout_period)

#     try:
#         # Wait for the result, letting process_request handle timeout behavior
#         result = future.result()  # Don't pass the timeout here
#         return jsonify(result), 200
#     except Exception as e:
#         return jsonify({"Response": "Fail", "error": str(e)}), 200
    

# Security_Layer.run(debug = True, port=4029, host= '0.0.0.0', use_reloader= False)

print(Prompt_Injection_Detector('hello','en',0.5))