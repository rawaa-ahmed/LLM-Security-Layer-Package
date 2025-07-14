# import warningsgit
# warnings.simplefilter('ignore')
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

class BinaryDataset(Dataset):
    """BinaryDataset class for encoding input data using the tokenizer before the DataLoader.
    
    Args:
        dataframe (DataFrame): input text dataset and labels
        tokenizer (transformer): A transformer tokenizer model to encode text data
        max_len (integer): Maximum length 

    Returns:
        Dict: ids (tensor), masks (tensor), token_type_ids (tensor), targets (tensor)
    """
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class AutoModelClass(torch.nn.Module):
    """AutoModelClass is for loading the auto models and adding 3 layers: pre_classifier (linear), dropout, classifier (linear)  

    Args:
        model_name (str): the model name 
        
    Returns:
        The loaded model
    """

    def __init__(self,model_name):
        super(AutoModelClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 1)
    # The forward function gets the initial output from the loaded model, then pass it through the additional layers and returns the final output.
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def validation(model,testing_loader):
    """ Validation function to perform calssification and output model results after training.
    
    Args:
        model : the trained model 
        testing loader: the test data loader
        
    Returns:
        list of list of float: the final output probabilities
    """
    # fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        # for _, data in enumerate(testing_loader, 0):
        _,data=next(enumerate(testing_loader))
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        # targets = data['targets'].to(device, dtype = torch.float)
        outputs = model(ids, mask, token_type_ids)
        # fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs #, fin_targets




Prompt_Injection_Detection_english_model_path = 'PI_Models/English/pytorch_model.bin'
Prompt_Injection_Detection_english_tokenizer_path = 'PI_Models/English/'

Prompt_Injection_Detection_arabic_model_path =  'PI_Models/Arabic/pytorch_model.bin'
Prompt_Injection_Detection_arabic_tokenizer_path =  'PI_Models/Arabic/'


Offencive_Content_english_model_path =  'OS_Models/English/Falconsai/offensive_speech_detection.bin'
Offencive_Content_english_tokenizer_path =  'OS_Models/English/Falconsai/'

Offencive_Content_arabic_model_path =  'OS_Models/Arabic/aubmindlab/bert-base-arabertv02-twitter.bin'
Offencive_Content_arabic_tokenizer_path =  'OS_Models/Arabic/aubmindlab/'



class Model:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, from_tf=True)
        self.model = torch.load(model_path)
        self.model.eval()
        self.model.to(device)
        
    def predict(self,prompt):
        # while(len(prompt)<128):
        #     prompt+=' ' + prompt
        # print(prompt)
        df=pd.DataFrame()
        df['text']=[prompt]
        df['labels']=0
        testing_set = BinaryDataset(df, self.tokenizer, 128)
        testing_loader = DataLoader(testing_set)
        outputs = validation(self.model, testing_loader)
        return outputs
    
        
en_pi_model=Model(Prompt_Injection_Detection_english_model_path,Prompt_Injection_Detection_english_tokenizer_path)
ar_pi_model=Model(Prompt_Injection_Detection_arabic_model_path,Prompt_Injection_Detection_arabic_tokenizer_path)
en_off_model=Model(Offencive_Content_english_model_path,Offencive_Content_english_tokenizer_path)
ar_off_model=Model(Offencive_Content_arabic_model_path,Offencive_Content_arabic_tokenizer_path)





def prompt_injection_detection(prompt = "", language="English", threshold=0.7):
    # handling prompt value
    if not(prompt.strip()):   
        raise Exception("Please enter a non empty string query")

    # handling language value
    if not((language == 'Arabic') or (language == 'English') or (language == None)) :   
        raise Exception("Please enter valid language [Arabic/English]")   
    # handling threshold value
    if threshold == None:
        threshold = 0.7
    if not(isinstance(threshold, float) and 0 <= threshold <= 1):
        raise Exception("Please enter valid threshold [0:1]")
    
    if language == 'Arabic':
        outputs = ar_pi_model.predict(prompt)
        #return {"score": outputs[0][0], "threshold": threshold, "language": language, "PID_output": 'safe'}
        if outputs[0][0] > threshold:
            return {"score": outputs[0][0], "threshold": threshold, "language": language, "PID_output": 'malicious'}    
        elif outputs[0][0] < threshold:
            return {"score": outputs[0][0], "threshold": threshold, "language": language, "PID_output": 'safe'}
        
    elif (language == 'English' or language == None):
        outputs = en_pi_model.predict(prompt)
        #return {"score": outputs[0][0], "threshold": threshold, "language": language, "PID_output": 'safe'} 
        if outputs[0][0] > threshold:
            return {"score": outputs[0][0], "threshold": threshold, "language": 'English', "PID_output": 'malicious'}
        elif outputs[0][0] < threshold:
            return {"score": outputs[0][0], "threshold": threshold, "language": 'English', "PID_output": 'safe'}                 
    else:
        raise Exception("Please enter valid input")
    
def offensive_content_detection(prompt = "", language="English", threshold=0.7):
    # handling prompt value
    if not(prompt.strip()):   
        raise Exception("Please enter a non empty string query")

    # handling language value
    if not((language == 'Arabic') or (language == 'English') or (language == None)) :   
        raise Exception("Please enter valid language [Arabic/English]")   
    # handling threshold value
    if threshold == None:
        threshold = 0.7
    if not(isinstance(threshold, float) and 0 <= threshold <= 1):
        raise Exception("Please enter valid threshold [0:1]")
    
    if language == 'Arabic':
        outputs = ar_off_model.predict(prompt)
        if outputs[0][0] > threshold:
            return {"score": outputs[0][0], "threshold": threshold, "language": language, "PID_output": 'offensive'}    
        elif outputs[0][0] < threshold:
            return {"score": outputs[0][0], "threshold": threshold, "language": language, "PID_output": 'safe'}
        
    elif (language == 'English' or language == None):
        outputs = en_off_model.predict(prompt)
        # print('\n',type(outputs[0]))
        if outputs[0][0] > threshold:
            return {"score": outputs[0][0], "threshold": threshold, "language": 'English', "PID_output": 'offensive'}
        elif outputs[0][0] < threshold:
            return {"score": outputs[0][0], "threshold": threshold, "language": 'English', "PID_output": 'safe'}                 
    else:
        raise Exception("Please enter valid input")
    
print(prompt_injection_detection('hello','English',0.5))