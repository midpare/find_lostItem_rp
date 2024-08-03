import boto3, cv2, json, random, math, asyncio, functools, pickle
import numpy as np

from sagemaker.pytorch import PyTorchPredictor
from sagemaker.deserializers import JSONDeserializer
runtime_client = boto3.client('sagemaker-runtime')

endpoints = {
    "detection": "pytorch-inference-yolov8x",
    "pose": "pytorch-inference-yolov8x-pose",
}

owns = dict()

def append_owns(result, owns):
    detection = result[0]
    pose = result[1]

    for i in range(len(detection)):
        for human in pose[i]:
            keypoints = human['keypoints']

            hands = (keypoints['x'][9], keypoints['y'][9]), (keypoints['x'][10], keypoints['y'][10])

            for object in detection[i]:
                if object['name'] == 'person':
                    continue

                if not ('track_id' in human.keys() and 'track_id' in object.keys()):
                    continue

                human_track_id = human['track_id']
                object_track_id = object['track_id']

                if not human_track_id in owns:
                    owns[human_track_id] = dict()
                box = object['box']
                object_pos = box['x1'], box['y1'], box['x2'], box['y2']

                center_pos = (object_pos[0] + object_pos[2]) / 2, (object_pos[1] + object_pos[3]) / 2
                distances = []
                flag = 0
                for hand_pos in hands:
                    if hand_pos == [0.0, 0.0]:
                        continue

                    d = math.sqrt((hand_pos[0] - center_pos[0]) ** 2 + (hand_pos[1] - center_pos[1]) ** 2)
                    distances.append(d)

                    if d < 100:
                        for hid in owns:
                            if (object_track_id, object['name']) in owns[hid].keys() and (not hid == human_track_id):
                                continue

                        owns[human_track_id][(object_track_id, object['name'])] = True
                        flag = 1
                
                for d in distances:
                    if flag == 0 and d > 700 and (object_track_id, object['name']) in owns[human_track_id]:
                        owns[human_track_id][(object_track_id, object['name'])] = False

async def inference(client, images, k):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, functools.partial(client.invoke_endpoint, 
        EndpointName=endpoints[k],
        ContentType='image/jpeg',
        Accept='application/json',
        Body=images
    ))    

    output = json.loads(response["Body"].read().decode('utf-8'))

    output = [json.loads(e) for e in output]

    return output

async def processing(images):
    images = pickle.dumps(images)
    tasks = [asyncio.create_task(inference(runtime_client, images, i)) for i in list(endpoints.keys())]
    result = await asyncio.gather(*tasks)

    append_owns(result, owns)

    return result, owns