import os
import math
import json

n=5

for filename in os.listdir('jsons'):
    file_path=os.path.join('jsons',filename)
    with open (file_path,'r',encoding='utf-8') as f:
        data=json.load(f)
        new_chunks=[]
        num_chunks=len(data['chunks'])
        num_groups=math.ceil(num_chunks/n)

        for i in range(num_groups):
            start_idx=i*n 
            end_idx=min((i+1)*n,num_chunks)

            chunk_group=data['chunks'][start_idx:end_idx]

            new_chunks.append({
                'number':data['chunks'][0]['number'],
                'title':chunk_group[0]['title'],
                'start':chunk_group[0]['start'],
                'end':chunk_group[-1]['end'],
                'text':" ".join(c['text'] for c in chunk_group)
            })
        
        #Save file without double .json
        os.makedirs('newjsons',exist_ok=True)
        with open(os.path.join('newjsons',filename),'w',encoding='utf-8') as json_file:
            json.dump({'chunks':new_chunks,'text':data['text']},json_file,indent=4)


#In this code we are merging every 5 chunks into one chunk and saving it in newjsons folder. We are doing this because we want to create embeddings for larger chunks of text rather than smaller chunks. This will help us to get better results when we search for similar chunks later.
# Problem with smaller chunks is that they may not contain enough context to get good embeddings and also they may not be relevant to the query. By merging them into larger chunks we can get better embeddings and also we can get more relevant results when we search for similar chunks later. 