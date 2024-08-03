import boto3, cv2, json, random, io, pickle, os
import process

runtime_client = boto3.client('sagemaker-runtime')

endpoints = {
    "detection": "pytorch-inference-yolov8x",
    "pose": "pytorch-inference-yolov8x-pose",
}

def getFiles(path):
    images = []
    labels = []
    data_path = os.listdir(path)

    for i in data_path:
        value_path = os.listdir('%s/%s'%(path, i))

        for j in value_path:
            
            if i == 'image':
                images.append(f'{path}/{i}/{j}')
            elif i == "label":
                labels.append(f'{path}/{i}/{j}')

            # labels.append(label)  
            # files.append(image)
    return images, labels
images, labels = getFiles('data')

success = 0
fail = 0
total = len(images)

for i in range(total):
    owns = {}
    print(i)
    image = cv2.imread(f'./{images[i]}')

    with open(labels[i]) as json_file:
        label = json.load(json_file)
    payload = cv2.imencode('.jpg', image)[1].tobytes()

    arr = [payload]

    arr = pickle.dumps(arr)

    result = []
    for i in endpoints:
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoints[i],
            ContentType='image/jpeg',
            Accept='application/json',
            Body=arr
        )

        output = json.loads(response["Body"].read().decode('utf-8'))

        output = [json.loads(e) for e in output]

        result.append(output)

    process.append_owns(result, owns)

    flag = 0
    for human_id in owns:
        for k in owns[human_id]:
            if k[1] in label and owns[human_id][k]:
                flag += 1

    if flag == len(label):
        success += 1
    else:
        fail += 1
    

print(f"Total: {total}, percent: {int((success / total) * 10000) / 100}%, asdf")
