import random
import base64
import json
import aiohttp
import asyncio

class infoForFirstTask:

  def __init__(self):
    self.session = aiohttp.ClientSession()

  def __del__(self):
    print("closing session")
    self.session.close()

  def find_files_with_extension(self, user_name, repo_name, extension, github_token, idx):
    headers = self.prepare_headers(github_token)

    params = {}
    if extension == '.md':
      # params = {"q": f'repo:{user_name}/{repo_name} filename:README path:/ extension:{extension}'}
      params = {"q": f'repo:{user_name}/{repo_name} extension:{extension}'}
    elif extension == '.ipynb':
      params = {"q": f'repo:{user_name}/{repo_name} extension:{extension}'}
    else:
      ValueError("No extension provided")

    
    url = f'https://api.github.com/search/code'
    res = requests.get(url, headers=headers, params=params)
    res.raise_for_status()

    # return list of tuples (file_name, path to file) 
    try:
      return [{'path': item['path'], 'id': idx} for item in res.json()["items"] ]
    except: 
      return []

  def prepare_headers(self, github_token): 

    headers = {}
    h1 = 'application/vnd.github.v3+json' 
    headers['Accept'] = h1
    if github_token:
        headers['Authorization'] = f'token {github_token}'

    return headers

  def github_read_file(self, user_name, repo_name, file_path, github_token):
    headers = self.prepare_headers(github_token)
    url = f'https://api.github.com/repos/{user_name}/{repo_name}/contents/{file_path}'
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    file_content = data['content']
    file_content_encoding = data.get('encoding')
    if file_content_encoding == 'base64':
      file_content = base64.b64decode(file_content).decode()

    # file_content = json.loads(file_content)
    return file_content

  def size_of_repo(self, user_name, repo_name, github_token):
    headers = self.prepare_headers(github_token)
    url = f'https://api.github.com/repos/{user_name}/{repo_name}'
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    # repository may not exist
    try:
      return int(res.json()['size'])
    except:
      return int(-1)

  async def as_size_of_repo(self, user_name, repo_name, github_token, idx):
    headers = self.prepare_headers(github_token)
    url = f'https://api.github.com/repos/{user_name}/{repo_name}'
    # async with self.session.get(url, headers=headers) as r:
    #   json_body = await r.json()
    res = await self.session.get(url, headers=headers)
    # res.raise_for_status()
    json_body = await res.json()
    # repository may not exist
    try:
      return {'size':int(json_body['size']), 'idx': idx}
    except:
      return {'size':int(-1), 'idx': idx}  
  
  async def as_find_files_in_repo(self, user_name, repo_name, github_token, idx, asyncio_semaphore, extension=None, file_name=None, path=None):
    async with asyncio_semaphore:
      headers = self.prepare_headers(github_token)

      if (extension is None) and (file_name is None) and (path is None):
        raise("atleast one from [extension, file_name, path] needs to be set")

      q = f'repo:{user_name}/{repo_name} '
      if extension is not None:
        q += f'extension:{extension} '
      if file_name is not None:
        q += f'filename:{file_name} '
      if path is not None:
        q += f'path:{path} '

      params= {"q": q}
      url = f'https://api.github.com/search/code'
      res = await self.session.get(url, headers=headers, params=params)
      async with res:
        if res.status != 200:
          print(f'res.status: {res.status}')
          raise('Check rate-limit for tokens (most probable reason)')

      json_body = await res.json()
      try:
        return [{'path': item['path'], 'id': idx} for item in json_body["items"] ]
      except: 
        return []  