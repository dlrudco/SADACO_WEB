from locale import str
import os
import json

import sadaco
from sadaco.pipelines.ICBHI import ICBHI_Basic_Trainer

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel

import datetime
import random
import string
import pickle

from print_format import Formatter
IMG_EXT = ['png', 'jpg', 'tif', 'tiff', 'jpeg', 'bmp']

FAIL_INTERNAL = 'Internal Process Failure : '
sessions  = {}

CONFIG_PATH = 'pipelines/configs'

DEFAULT_CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits + '@$^_'
class Session:
    def __init__(self, config_index):
        now = datetime.datetime.now()
        today = now.strftime('%Y%m%d%H%M%S')
        self.token = today+self.id_generator()
        global sessions
        sessions[self.token] = self
        self.config = get_config(get_configs_list()[config_index])
        self.formatter = Formatter()
        self.trainer = ICBHI_Basic_Trainer(self.config)
        self.progress = -1
        
    def __str__(self):
        status = f'I am session {self.token}\n'
        status += f'My configs are \n{self.formatter(self.config)}' 
        return status
        
    def dummy_update(self):
        from tqdm import tqdm
        import time
        for i in tqdm(range(100)):
            time.sleep(1)
            self.progress = i
        return i
    
    def train_model(self, background_tasks:BackgroundTasks):
        background_tasks.add_task(self.trainer.train)
        # background_tasks.add_task(self.dummy_update)
        return {"message" : 'Started Training'}
    
    def get_logs(self):
        if self.config.use_wandb:
            return {"Redirect" : self.trainer.logger.url}
        else:
            return json.load(open(os.path.join(self.trainer.logger.log_path, 'configs.json'), 'rb'))
    
    def current_train_progress(self):
        self.progress = self.trainer._progress
        return {"message" : f'Currently done running {self.progress+1} epochs'}
        
        
    @staticmethod
    def id_generator(size=12, chars=DEFAULT_CHARS):
        return ''.join(random.choice(chars) for _ in range(size))        
        
app = FastAPI()
# app.mount("/", StaticFiles(directory="web/asset"), name="static")
        
@app.get("/")
async def hello_world():
    return{'hello':'world'}

@app.get("/register")
async def new_session(config_index):
    my = Session(config_index=int(config_index))
    return my.token

@app.get("/sessions")
async def print_session_list():
    return list(sessions.keys())

@app.get("/print_session")
async def print_session(token):
    print(sessions[token].__str__())
    return sessions[token].__str__()

@app.get("/start_train")
async def train(token, background_tasks: BackgroundTasks):
    return sessions[token].train_model(background_tasks)
    
@app.get("/current_progress")
async def get_progress(token):
    return sessions[token].current_progress()

@app.get("/readme")
async def readme():
    text = '''
    In order to provide stateful services, we create session object with unique random token
    to each user who calls 'register'. A user can also choose to jump btw sessions with tokens
    '''
    return text
        
@app.get('/get_configs')
def get_configs_list():
    file_list = sadaco.utils.web_utils.get_configs()
    config_list = sorted(file_list)
    return config_list

@app.get('/choose_config/{name}')
def get_config(name):
    configs = sadaco.utils.web_utils.load_config(name)
    return configs


@app.get("/vector_image", responses={200: {"description": "A picture of a vector image.", "content" : {"image/jpeg" : {"example" : "No example available. Just imagine a picture of a vector image."}}}})
def image_endpoint():
    file_path = '/home/ncl/kclee/IDX/sadaco/sadaco/demo/demo3_gradcam/GradCAM-Both.png'
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png", filename="vector_image_for_you.jpg")
    return {"error" : "File not found!"}


if __name__ == "__main__":
    uvicorn.run("RESTful:app", host='0.0.0.0', port=7942, reload=True)